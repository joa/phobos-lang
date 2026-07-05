use melior::ir::Module;
use mlir_sys::{
    mlirBlockGetFirstOperation, mlirIdentifierStr, mlirModuleGetBody, mlirOperationDestroy,
    mlirOperationGetFirstRegion, mlirOperationGetName, mlirOperationGetNextInBlock,
    mlirOperationMoveBefore, mlirOperationRemoveFromParent, mlirRegionGetFirstBlock,
};

pub unsafe fn flatten_gpu_modules(module: &mut Module) {
    let body = unsafe { mlirModuleGetBody(module.to_raw()) };
    let mut op = unsafe { mlirBlockGetFirstOperation(body) };

    while !op.ptr.is_null() {
        let next = unsafe { mlirOperationGetNextInBlock(op) };

        let sref = unsafe { mlirIdentifierStr(mlirOperationGetName(op)) };
        let name = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(
                sref.data as *const u8,
                sref.length,
            ))
            .unwrap_or("")
        };

        if name == "gpu.module" {
            let region = unsafe { mlirOperationGetFirstRegion(op) };
            let block = unsafe { mlirRegionGetFirstBlock(region) };

            loop {
                let inner = unsafe { mlirBlockGetFirstOperation(block) };
                if inner.ptr.is_null() {
                    break;
                }
                unsafe { mlirOperationMoveBefore(inner, op) };
            }

            unsafe { mlirOperationRemoveFromParent(op) };
            unsafe { mlirOperationDestroy(op) };
        }

        op = next;
    }
}
