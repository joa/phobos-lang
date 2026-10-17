//! Free device memory as the whole card sees it, and the card's power draw
//! and temperature, through NVML.
//!
//! `cuMemGetInfo` answers for this process alone. Under WDDM it reports a
//! budget less what this process holds, and memory another process takes
//! does not lower it, so a cache sized from it never learns that the card
//! filled up around it. NVML counts every process, as `nvidia-smi` does.
//! The library ships with the driver and is loaded at first use, so a
//! machine without it falls back to `cuMemGetInfo`.

use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use cust::sys::{CUresult, cuDeviceGetPCIBusId};
use libloading::{Library, Symbol};

#[repr(C)]
struct Memory {
    total: u64,
    free: u64,
    used: u64,
}

type Init = unsafe extern "C" fn() -> c_int;
type HandleByBusId = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_int;
type MemoryInfo = unsafe extern "C" fn(*mut c_void, *mut Memory) -> c_int;
type Reading = unsafe extern "C" fn(*mut c_void, *mut u32) -> c_int;
type Temperature = unsafe extern "C" fn(*mut c_void, c_int, *mut u32) -> c_int;

/// `NVML_TEMPERATURE_GPU`, the die's sensor.
const TEMPERATURE_GPU: c_int = 0;

struct Nvml {
    // Keeps the symbols below loaded.
    _lib: Library,
    device: *mut c_void,
    memory_info: MemoryInfo,
    // An older driver may lack these; free memory still works without them.
    power_usage: Option<Reading>,
    power_limit: Option<Reading>,
    temperature: Option<Temperature>,
}

// The handle is an opaque id NVML hands out once, valid from any thread.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

const LIBRARY: &str = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };

impl Nvml {
    /// NVML's handle for CUDA device 0, matched by PCI bus id since the two
    /// libraries may number the devices differently.
    fn open() -> Option<Self> {
        let mut bus_id = [0 as c_char; 32];
        let got = unsafe { cuDeviceGetPCIBusId(bus_id.as_mut_ptr(), bus_id.len() as c_int, 0) };
        if got != CUresult::CUDA_SUCCESS {
            return None;
        }
        unsafe {
            let lib = Library::new(LIBRARY).ok()?;
            let init: Symbol<Init> = lib.get(b"nvmlInit_v2\0").ok()?;
            let by_bus_id: Symbol<HandleByBusId> = lib.get(b"nvmlDeviceGetHandleByPciBusId_v2\0").ok()?;
            let memory_info = *lib.get::<MemoryInfo>(b"nvmlDeviceGetMemoryInfo\0").ok()?;
            let power_usage = lib.get::<Reading>(b"nvmlDeviceGetPowerUsage\0").ok().map(|f| *f);
            let power_limit = lib.get::<Reading>(b"nvmlDeviceGetEnforcedPowerLimit\0").ok().map(|f| *f);
            let temperature = lib.get::<Temperature>(b"nvmlDeviceGetTemperature\0").ok().map(|f| *f);
            if init() != 0 {
                return None;
            }
            let mut device = std::ptr::null_mut();
            if by_bus_id(bus_id.as_ptr(), &mut device) != 0 {
                return None;
            }
            Some(Self {
                _lib: lib,
                device,
                memory_info,
                power_usage,
                power_limit,
                temperature,
            })
        }
    }

    fn get() -> Option<&'static Nvml> {
        static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
        NVML.get_or_init(Nvml::open).as_ref()
    }
}

/// What the card's sensors read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardSensors {
    pub power_milliwatts: u32,
    /// The draw the driver holds the card to, when it says.
    pub power_limit_milliwatts: Option<u32>,
    pub temperature_celsius: u32,
}

/// The power draw and die temperature of CUDA device 0, or `None` when NVML
/// or either reading is missing. Cheap enough to poll a few times a second.
pub fn card_sensors() -> Option<CardSensors> {
    let nvml = Nvml::get()?;
    let read = |f: Option<Reading>| {
        let mut value = 0u32;
        let got = unsafe { f?(nvml.device, &mut value) };
        (got == 0).then_some(value)
    };
    let mut temperature_celsius = 0u32;
    let got = unsafe { (nvml.temperature?)(nvml.device, TEMPERATURE_GPU, &mut temperature_celsius) };
    if got != 0 {
        return None;
    }
    Some(CardSensors {
        power_milliwatts: read(nvml.power_usage)?,
        power_limit_milliwatts: read(nvml.power_limit),
        temperature_celsius,
    })
}

/// Free bytes on CUDA device 0 across every process, or `None` when NVML is
/// missing. Needs a current CUDA context the first time.
///
/// Under WDDM this does not fall to zero: once the card is over-subscribed
/// the driver keeps a few hundred MiB unoccupied and pages instead, so a
/// reading near that floor means "full", not "that much left".
pub fn card_free_bytes() -> Option<usize> {
    let nvml = Nvml::get()?;
    let mut memory = Memory { total: 0, free: 0, used: 0 };
    let got = unsafe { (nvml.memory_info)(nvml.device, &mut memory) };
    (got == 0).then_some(memory.free as usize)
}
