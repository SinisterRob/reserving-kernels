use std::ffi::{c_char, c_void};
use std::ptr::{null, null_mut};
use std::sync::Mutex;

use reserving_kernels::Triangle;

const XLTYPE_NUM: u32 = 0x0001;
const XLTYPE_STR: u32 = 0x0002;
const XLTYPE_ERR: u32 = 0x0010;
const XLTYPE_MULTI: u32 = 0x0040;
const XLTYPE_MISSING: u32 = 0x0080;
const XLTYPE_NIL: u32 = 0x0100;
const XLTYPE_INT: u32 = 0x0800;
const XL_BIT_XL_FREE: u32 = 0x1000;
const XL_BIT_DLL_FREE: u32 = 0x4000;

pub const XLERR_VALUE: i32 = 15;
pub const XLERR_NUM: i32 = 36;

const XL_FREE: i32 = 0x4000;
const XL_GET_NAME: i32 = 0x4009;
const XLF_SET_NAME: i32 = 88;
const XLF_REGISTER: i32 = 149;
const XLF_UNREGISTER: i32 = 201;
const MAX_ORIGINS: i32 = 512;

// The layout and callback signature come from XLCALL.H and XLCALL.CPP in the
// Microsoft Excel XLL SDK. Only the value variants used by this adapter are named.
#[repr(C)]
pub struct Xloper12 {
    value: XlValue,
    kind: u32,
}

#[repr(C)]
union XlValue {
    number: f64,
    integer: i32,
    string: *mut u16,
    array: XlArray,
    _storage: [u64; 3],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct XlArray {
    cells: *mut Xloper12,
    rows: i32,
    cols: i32,
}

const _: () = {
    assert!(size_of::<Xloper12>() == 32);
    assert!(align_of::<Xloper12>() == 8);
    assert!(std::mem::offset_of!(Xloper12, kind) == 24);
};

impl Xloper12 {
    fn number(number: f64) -> Self {
        Self {
            value: XlValue { number },
            kind: XLTYPE_NUM,
        }
    }

    fn nil() -> Self {
        Self {
            value: XlValue { integer: 0 },
            kind: XLTYPE_NIL,
        }
    }

    fn kind(&self) -> u32 {
        self.kind & !(XL_BIT_XL_FREE | XL_BIT_DLL_FREE)
    }

    fn read_number(&self) -> Result<f64, i32> {
        match self.kind() {
            // SAFETY: Each tag selects the corresponding union member.
            XLTYPE_NUM => Ok(unsafe { self.value.number }),
            // SAFETY: xltypeInt and xltypeErr both use an i32 value.
            XLTYPE_INT => Ok(f64::from(unsafe { self.value.integer })),
            // SAFETY: xltypeErr uses the integer member for the error code.
            XLTYPE_ERR => Err(unsafe { self.value.integer }),
            _ => Err(XLERR_VALUE),
        }
    }

    pub fn return_array<const COLS: usize>(rows: Vec<[f64; COLS]>) -> *mut Self {
        let row_count = rows.len() as i32;
        let cells: Box<[_]> = rows.into_iter().flatten().map(Self::number).collect();
        Box::into_raw(Box::new(Self {
            value: XlValue {
                array: XlArray {
                    cells: Box::into_raw(cells).cast(),
                    rows: row_count,
                    cols: COLS as i32,
                },
            },
            kind: XLTYPE_MULTI | XL_BIT_DLL_FREE,
        }))
    }

    pub fn return_error(error: i32) -> *mut Self {
        Box::into_raw(Box::new(Self {
            value: XlValue { integer: error },
            kind: XLTYPE_ERR | XL_BIT_DLL_FREE,
        }))
    }
}

#[unsafe(no_mangle)]
unsafe extern "system" fn xlAutoFree12(result: *mut Xloper12) {
    if result.is_null() {
        return;
    }
    // SAFETY: Excel calls this once for each value returned with xlbitDLLFree.
    let result = unsafe { Box::from_raw(result) };
    if result.kind() == XLTYPE_MULTI {
        // SAFETY: Our worksheet functions only return arrays of numeric cells. The
        // pointer and dimensions are those from the boxed slice in return_array.
        let array = unsafe { result.value.array };
        let cells =
            std::ptr::slice_from_raw_parts_mut(array.cells, (array.rows * array.cols) as usize);
        // SAFETY: The original slice allocation belongs to this result alone.
        drop(unsafe { Box::from_raw(cells) });
    }
}

pub unsafe fn triangle(input: *const Xloper12) -> Result<Triangle, i32> {
    // SAFETY: Excel gives the worksheet function a borrowed, tagged XLOPER12.
    let input = unsafe { input.as_ref() }.ok_or(XLERR_VALUE)?;
    if input.kind() != XLTYPE_MULTI {
        return Err(input.read_number().err().unwrap_or(XLERR_VALUE));
    }
    // SAFETY: xltypeMulti selects the array member.
    let array = unsafe { input.value.array };
    if array.rows != array.cols || !(2..=MAX_ORIGINS).contains(&array.rows) || array.cells.is_null()
    {
        return Err(XLERR_VALUE);
    }
    let n = array.rows as usize;
    // SAFETY: Excel provides rows * cols initialized cells, valid until this call returns.
    let cells = unsafe { std::slice::from_raw_parts(array.cells, n * n) };
    let mut data = Vec::with_capacity(cells.len());
    for (row, cells) in cells.chunks_exact(n).enumerate() {
        let (observed, future) = cells.split_at(n - row);
        for cell in observed {
            data.push(cell.read_number()?);
        }
        if !future.iter().all(|cell| cell.kind() == XLTYPE_NIL) {
            return Err(XLERR_VALUE);
        }
        data.resize((row + 1) * n, f64::NAN);
    }
    Triangle::try_new(n, n, data).map_err(|_| XLERR_NUM)
}

pub unsafe fn tail_factor(input: *const Xloper12) -> Result<f64, i32> {
    // SAFETY: Excel gives the worksheet function a borrowed, tagged XLOPER12.
    let input = unsafe { input.as_ref() }.ok_or(XLERR_VALUE)?;
    if matches!(input.kind(), XLTYPE_MISSING | XLTYPE_NIL) {
        return Ok(1.0);
    }
    let tail = input.read_number()?;
    if tail.is_finite() && tail >= 1.0 {
        Ok(tail)
    } else {
        Err(XLERR_NUM)
    }
}

type ProcAddress = unsafe extern "system" fn();
type ExcelCallback = unsafe extern "system" fn(i32, i32, *mut *mut Xloper12, *mut Xloper12) -> i32;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> Option<ProcAddress>;
}

fn excel_callback() -> Option<ExcelCallback> {
    // SAFETY: A null name requests the process executable module handle.
    let module = unsafe { GetModuleHandleW(null()) };
    if module.is_null() {
        return None;
    }
    // SAFETY: The executable handle is live and the C string is NUL terminated.
    let address = unsafe { GetProcAddress(module, c"MdCallBack12".as_ptr()) }?;
    // SAFETY: MdCallBack12 has the signature specified by the Excel XLL SDK.
    Some(unsafe { std::mem::transmute::<ProcAddress, ExcelCallback>(address) })
}

fn counted_string(text: &str) -> Vec<u16> {
    let mut units: Vec<u16> = std::iter::once(0).chain(text.encode_utf16()).collect();
    units[0] = (units.len() - 1) as u16;
    units
}

fn string_arg(text: &mut [u16]) -> Xloper12 {
    Xloper12 {
        value: XlValue {
            string: text.as_mut_ptr(),
        },
        kind: XLTYPE_STR,
    }
}

#[cfg(test)]
mod tests;

struct Function {
    export: &'static str,
    name: &'static str,
}

const FUNCTIONS: [Function; 2] = [
    Function {
        export: "rk_xll_chain_ladder",
        name: "RK.CHAIN_LADDER",
    },
    Function {
        export: "rk_xll_mack",
        name: "RK.MACK",
    },
];

static REGISTRATIONS: Mutex<[Option<f64>; FUNCTIONS.len()]> = Mutex::new([None; FUNCTIONS.len()]);

fn register(call: ExcelCallback, module: &mut Xloper12, function: &Function) -> Option<f64> {
    let mut text = [
        function.export,
        "QQQ$",
        function.name,
        "triangle,tail",
        "Reserving Kernels",
    ]
    .map(counted_string);
    let [export, signature, name, arguments, category] =
        text.each_mut().map(|value| string_arg(value));
    let mut args = [
        export,
        signature,
        name,
        arguments,
        Xloper12::number(1.0),
        category,
    ];
    let [export, signature, name, arguments, macro_type, category] = args.each_mut();
    let mut pointers = [
        module, export, signature, name, arguments, macro_type, category,
    ]
    .map(std::ptr::from_mut);
    let mut result = Xloper12::nil();
    // SAFETY: Argument pointers and counted strings are live for the entire callback.
    let status = unsafe {
        call(
            XLF_REGISTER,
            pointers.len() as i32,
            pointers.as_mut_ptr(),
            &mut result,
        )
    };
    (status == 0 && result.kind() == XLTYPE_NUM)
        .then(|| result.read_number().ok())
        .flatten()
}

fn unregister(call: ExcelCallback, ids: &mut [Option<f64>; FUNCTIONS.len()]) {
    for (function, id) in FUNCTIONS.iter().zip(ids) {
        if let Some(id) = id.take() {
            let mut text = counted_string(function.name);
            let mut name = string_arg(&mut text);
            let mut number = Xloper12::number(id);
            // SAFETY: Each callback takes one live argument. A null result discards the return.
            unsafe {
                call(
                    XLF_SET_NAME,
                    1,
                    &mut std::ptr::from_mut(&mut name),
                    null_mut(),
                );
                call(
                    XLF_UNREGISTER,
                    1,
                    &mut std::ptr::from_mut(&mut number),
                    null_mut(),
                );
            }
        }
    }
}

#[unsafe(no_mangle)]
extern "system" fn xlAutoOpen() -> i32 {
    let Some(call) = excel_callback() else {
        return 0;
    };
    let mut ids = REGISTRATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if ids.iter().all(Option::is_some) {
        return 1;
    }
    let mut module = Xloper12::nil();
    // SAFETY: Excel calls xlAutoOpen on its main thread. xlGetName takes no arguments.
    if unsafe { call(XL_GET_NAME, 0, null_mut(), &mut module) } != 0 {
        return 0;
    }
    if module.kind() == XLTYPE_STR {
        for (id, function) in ids.iter_mut().zip(&FUNCTIONS) {
            *id = register(call, &mut module, function);
            if id.is_none() {
                break;
            }
        }
    }
    // SAFETY: xlFree releases the Excel-owned result of the successful xlGetName call.
    unsafe { call(XL_FREE, 1, &mut std::ptr::from_mut(&mut module), null_mut()) };
    if ids.iter().all(Option::is_some) {
        1
    } else {
        unregister(call, &mut ids);
        0
    }
}

#[unsafe(no_mangle)]
extern "system" fn xlAutoClose() -> i32 {
    if let Some(call) = excel_callback() {
        let mut ids = REGISTRATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        unregister(call, &mut ids);
    }
    1
}
