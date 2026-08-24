use std::sync::Once;

use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationLike},
    pass::PassManager,
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
};

use inkwell::targets::{InitializationConfig, Target};

mod mlir_flatten;

/// `register_all_passes` and `Target::initialize_nvptx` write into
/// process-global LLVM tables and aren't documented safe to race, unlike the
/// rest of [`gen_code`], which only touches its own `Context`. Run once per
/// process.
fn ensure_global_init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        register_all_passes();
        Target::initialize_nvptx(&InitializationConfig::default());
    });
}

/// Generate PTX for a given function body F.
///
/// A thin wrapper over [`gen_code`] from before there was more than one thing
/// to generate; the callers that only ever wanted NVIDIA keep using it.
pub fn gen_ptx<'p, F>(ctx: &'p phobos_base::context::Context, body: F) -> anyhow::Result<String>
where
    F: for<'c> FnOnce(
        &'p phobos_base::context::Context,
        &'c Context,
        &Module<'c>,
    ) -> anyhow::Result<()>,
{
    gen_code(ctx, body)
}

/// Generate the target's code for a given function body F.
pub fn gen_code<'p, F>(ctx: &'p phobos_base::context::Context, body: F) -> anyhow::Result<String>
where
    F: for<'c> FnOnce(
        &'p phobos_base::context::Context,
        &'c Context,
        &Module<'c>,
    ) -> anyhow::Result<()>,
{
    ensure_global_init();

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);

    let loc = Location::unknown(&context);
    let module = Module::new(loc);

    body(ctx, &context, &module)?;

    if ctx.print_phases {
        println!("=== MLIR ==========================");
        println!("{}", module.as_operation());
        println!("===================================");
    }

    lower_mlir_to_ptx(ctx, &context, module)
}

/// Lowering of MLIR to the target's code.
pub fn lower_mlir_to_ptx<'c>(
    ctx: &phobos_base::context::Context,
    mlir_ctx: &'c Context,
    mut module: Module<'c>,
) -> anyhow::Result<String> {
    let backend = ctx.gpu_config.backend();

    // 1) Lower the GPU dialect to LLVM
    let pm = PassManager::new(mlir_ctx);
    let top_raw = pm.as_operation_pass_manager().to_raw();

    unsafe {
        let pipeline = std::ffi::CString::new(backend.pipeline(ctx.index_bitwidth))?;

        let pipeline_ref = mlir_sys::mlirStringRefCreateFromCString(pipeline.as_ptr());

        unsafe extern "C" fn error_cb(
            msg: mlir_sys::MlirStringRef,
            _user_data: *mut std::ffi::c_void,
        ) {
            let s = unsafe { std::slice::from_raw_parts(msg.data as *const u8, msg.length) };
            eprintln!("[mlir pipeline error] {}", String::from_utf8_lossy(s));
        }

        let result = mlir_sys::mlirParsePassPipeline(
            top_raw,
            pipeline_ref,
            Some(error_cb),
            std::ptr::null_mut(),
        );

        if result.value == 0 {
            return Err(anyhow::anyhow!("Failed to construct MLIR pipeline"));
        }
    }

    pm.run(&mut module)
        .map_err(|_| anyhow::anyhow!("MLIR GPU lowering pass failed"))?;

    if ctx.print_phases {
        println!("=== MLIR LOWERING =================");
        println!("{}", module.as_operation());
        println!("===================================");
    }

    if !module.as_operation().verify() {
        return Err(anyhow::anyhow!(
            "MLIR module verification failed after GPU lowering"
        ));
    }

    // 2) Hoist llvm.func ops out of the gpu.module wrapper and into the
    //    top-level module, then destroy the empty gpu.module.
    unsafe { mlir_flatten::flatten_gpu_modules(&mut module) };

    // 3) Translate MLIR to LLVM IR
    let ir_text: Vec<u8> = unsafe {
        unsafe extern "C" {
            fn free(ptr: *mut std::ffi::c_void);
        }
        let raw_mlir_op = module.as_operation().to_raw();
        let ir_ptr = mlir_sys::mlirTranslateModuleToLLVMIRToString(raw_mlir_op);
        if ir_ptr.is_null() {
            return Err(anyhow::anyhow!("MLIR to LLVM IR translation failed"));
        }
        let cstr = std::ffi::CStr::from_ptr(ir_ptr);
        let bytes = cstr.to_bytes_with_nul().to_vec();
        free(ir_ptr as *mut _);
        bytes
    };

    let ir_str: String = String::from_utf8(ir_text.clone())?;

    if ctx.print_phases {
        println!("=== LLVM IR =======================");
        println!("{}", ir_str);
        println!("===================================");
    }

    // 4) Compile LLVM IR to PTX. Target registration happened in
    // `ensure_global_init`, before `gen_code` reached here.
    let target = Target::from_name("nvptx64").ok_or_else(|| {
        anyhow::anyhow!("NVPTX target not found (must compile LLVM with NVPTX support)")
    })?;

    let llvm_ctx = inkwell::context::Context::create();
    let mem_buf =
        inkwell::memory_buffer::MemoryBuffer::create_from_memory_range_copy(&ir_text, "nvvm_ir");
    let llvm_mod = llvm_ctx
        .create_module_from_ir(mem_buf)
        .map_err(|e| anyhow::anyhow!("LLVM IR parse failed: {e}"))?;

    let machine = backend.llvm_target();
    let target_machine = target
        .create_target_machine(
            &inkwell::targets::TargetTriple::create(machine.triple),
            machine.cpu,
            machine.features,
            inkwell::OptimizationLevel::Aggressive,
            inkwell::targets::RelocMode::Default,
            inkwell::targets::CodeModel::Default,
        )
        .ok_or_else(|| anyhow::anyhow!("Failed to create target machine"))?;

    llvm_mod.set_data_layout(&target_machine.get_target_data().get_data_layout());
    llvm_mod.set_triple(&target_machine.get_triple());

    let ptx_buf = target_machine
        .write_to_memory_buffer(&llvm_mod, inkwell::targets::FileType::Assembly)
        .map_err(|e| anyhow::anyhow!("PTX codegen failed: {e}"))?;

    Ok(String::from_utf8_lossy(ptx_buf.as_slice())
        .trim_end_matches('\0')
        .to_owned())
}
