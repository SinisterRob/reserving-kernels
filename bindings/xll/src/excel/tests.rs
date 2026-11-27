use super::*;
use crate::{rk_xll_chain_ladder, rk_xll_mack};

fn sample() -> [Xloper12; 16] {
    [
        Some(100.0),
        Some(160.0),
        Some(195.0),
        Some(205.0),
        Some(120.0),
        Some(190.0),
        Some(232.0),
        None,
        Some(135.0),
        Some(218.0),
        None,
        None,
        Some(150.0),
        None,
        None,
        None,
    ]
    .map(|value| value.map_or_else(Xloper12::nil, Xloper12::number))
}

fn range(cells: &mut [Xloper12; 16]) -> Xloper12 {
    Xloper12 {
        value: XlValue {
            array: XlArray {
                cells: cells.as_mut_ptr(),
                rows: 4,
                cols: 4,
            },
        },
        kind: XLTYPE_MULTI,
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs {expected}"
    );
}

unsafe fn copy_result(result: *mut Xloper12, columns: i32) -> Vec<f64> {
    // SAFETY: The caller supplies a live worksheet result from the XLL.
    let result_ref = unsafe { &*result };
    assert_eq!(result_ref.kind, XLTYPE_MULTI | XL_BIT_DLL_FREE);
    // SAFETY: The checked tag selects the array member.
    let array = unsafe { result_ref.value.array };
    assert_eq!((array.rows, array.cols), (5, columns));
    // SAFETY: The XLL allocated this array with the dimensions returned here.
    let cells =
        unsafe { std::slice::from_raw_parts(array.cells, (array.rows * array.cols) as usize) };
    let numbers = cells
        .iter()
        .map(|cell| cell.read_number().unwrap())
        .collect();
    // SAFETY: The XLL gave the caller ownership of this result.
    unsafe { xlAutoFree12(result) };
    numbers
}

#[test]
fn worksheet_results_include_origin_rows_and_totals() {
    let mut cells = sample();
    let input = range(&mut cells);
    let tail = Xloper12 {
        kind: XLTYPE_MISSING,
        ..Xloper12::nil()
    };
    // SAFETY: The range, its cells, and tail are live and have valid tags.
    let (chain_ladder, mack) = unsafe {
        (
            copy_result(rk_xll_chain_ladder(&input, &tail), 2),
            copy_result(rk_xll_mack(&input, &tail), 3),
        )
    };
    assert_eq!(&chain_ladder[..2], &[205.0, 0.0]);
    for (chain_ladder, mack) in chain_ladder.chunks_exact(2).zip(mack.chunks_exact(3)) {
        assert_close(chain_ladder[0], mack[0], 1e-9);
        assert_close(chain_ladder[1], mack[1], 1e-9);
    }
    assert_close(chain_ladder[9], 231.31, 0.01);
    assert_close(mack[14], 3.41, 0.01);
}

#[test]
fn worksheet_errors_are_excel_errors_and_can_be_freed() {
    let mut cells = sample();
    let invalid_tail = Xloper12::number(0.5);
    cells[7] = Xloper12::number(0.0);
    let malformed = range(&mut cells);
    // SAFETY: All arguments are live. The first call fails at tail validation.
    unsafe {
        let mut valid_cells = sample();
        let valid = range(&mut valid_cells);
        let error = rk_xll_chain_ladder(&valid, &invalid_tail);
        assert_eq!((*error).read_number(), Err(XLERR_NUM));
        xlAutoFree12(error);

        let error = rk_xll_mack(&malformed, &Xloper12::nil());
        assert_eq!((*error).read_number(), Err(XLERR_VALUE));
        xlAutoFree12(error);

        let mut cells = sample();
        cells[0] = Xloper12 {
            value: XlValue { integer: 42 },
            kind: XLTYPE_ERR,
        };
        let error = rk_xll_mack(&range(&mut cells), &Xloper12::nil());
        assert_eq!((*error).read_number(), Err(42));
        xlAutoFree12(error);
    }
}
