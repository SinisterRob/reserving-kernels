//! Canonical published P&C actuarial benchmark datasets.
//!
//! All datasets are compiled directly into the crate as static `f64` arrays
//! with zero external I/O, making them immediately available in native tests,
//! examples, and `wasm32` targets.

use serde::{Deserialize, Serialize};

use crate::types::Triangle;

// Invariant for every `Triangle::new` call in this module: each dataset is a
// fixed-size `n × n` literal (shape asserted below in `tests`), observed cells
// are finite and non-negative, so it would pass `Triangle::try_new`. The
// unchecked crate-internal constructor is used only to keep these accessors
// infallible; `tests::datasets_pass_try_new_validation` enforces the contract.

const NAN: f64 = f64::NAN;

/// Paired cumulative paid and incurred triangles (e.g., Quarg & Mack 2004).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaidIncurredDataset {
    /// Cumulative paid loss triangle.
    pub paid: Triangle,
    /// Cumulative incurred (reported) loss triangle.
    pub incurred: Triangle,
}

/// Friedland (2010) US Industry Auto dataset (paid, reported, and earned premium).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FriedlandAutoDataset {
    /// Cumulative paid claims triangle (10×10).
    pub paid: Triangle,
    /// Cumulative reported claims triangle (10×10).
    pub reported: Triangle,
    /// Earned premium per origin year (length 10).
    pub earned_premium: Vec<f64>,
}

/// Reinsurance Association of America (RAA, 1991) 10×10 Automatic Facultative
/// General Liability incurred triangle from Mack (1993), *ASTIN Bulletin* 23(2).
pub fn raa() -> Triangle {
    #[rustfmt::skip]
    let data = vec![
        5012.0,  8269.0, 10907.0, 11805.0, 13539.0, 16181.0, 18009.0, 18608.0, 18662.0, 18834.0,
         106.0,  4285.0,  5396.0, 10666.0, 13782.0, 15599.0, 15496.0, 16169.0, 16704.0,     NAN,
        3410.0,  8992.0, 13873.0, 16141.0, 18735.0, 22214.0, 22863.0, 23466.0,     NAN,     NAN,
        5655.0, 11555.0, 15766.0, 21266.0, 23425.0, 26083.0, 27067.0,     NAN,     NAN,     NAN,
        1092.0,  9565.0, 15836.0, 22169.0, 25955.0, 26180.0,     NAN,     NAN,     NAN,     NAN,
        1513.0,  6445.0, 11702.0, 12935.0, 15852.0,     NAN,     NAN,     NAN,     NAN,     NAN,
         557.0,  4020.0, 10946.0, 12314.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
        1351.0,  6947.0, 13112.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
        3133.0,  5395.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
        2063.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
    ];
    Triangle::new(10, 10, data)
}

/// Taylor & Ashe (1983) / England & Verrall (1999, 2002) `GenIns` 10×10
/// cumulative paid loss triangle.
pub fn genins() -> Triangle {
    #[rustfmt::skip]
    let data = vec![
        357848.0, 1124788.0, 1735330.0, 2218270.0, 2745596.0, 3319994.0, 3466336.0, 3606286.0, 3833515.0, 3901463.0,
        352118.0, 1236139.0, 2170033.0, 3353322.0, 3799067.0, 4120063.0, 4647867.0, 4914039.0, 5339085.0,       NAN,
        290507.0, 1292306.0, 2218525.0, 3235179.0, 3985995.0, 4132918.0, 4628910.0, 4909315.0,       NAN,       NAN,
        310608.0, 1418858.0, 2195047.0, 3757447.0, 4029929.0, 4381982.0, 4588268.0,       NAN,       NAN,       NAN,
        443160.0, 1136350.0, 2128333.0, 2897821.0, 3402672.0, 3873311.0,       NAN,       NAN,       NAN,       NAN,
        396132.0, 1333217.0, 2180715.0, 2985752.0, 3691712.0,       NAN,       NAN,       NAN,       NAN,       NAN,
        440832.0, 1288463.0, 2419861.0, 3483130.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
        359480.0, 1421128.0, 2864498.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
        376686.0, 1363294.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
        344014.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
    ];
    Triangle::new(10, 10, data)
}

/// Quarg & Mack (2004) 7×7 cumulative paid and incurred triangles from the
/// original Munich Chain Ladder paper (*Blätter der DGVFM* / CAS E-Forum).
pub fn quarg_mack_2004() -> PaidIncurredDataset {
    #[rustfmt::skip]
    let paid = vec![
         576.0, 1804.0, 1970.0, 2024.0, 2074.0, 2102.0, 2131.0,
         866.0, 1948.0, 2162.0, 2232.0, 2284.0, 2348.0,    NAN,
        1412.0, 3758.0, 4252.0, 4416.0, 4494.0,    NAN,    NAN,
        2286.0, 5292.0, 5724.0, 5850.0,    NAN,    NAN,    NAN,
        1868.0, 3778.0, 4648.0,    NAN,    NAN,    NAN,    NAN,
        1442.0, 4010.0,    NAN,    NAN,    NAN,    NAN,    NAN,
        2044.0,    NAN,    NAN,    NAN,    NAN,    NAN,    NAN,
    ];
    #[rustfmt::skip]
    let incurred = vec![
         978.0, 2104.0, 2134.0, 2144.0, 2174.0, 2182.0, 2174.0,
        1844.0, 2552.0, 2466.0, 2480.0, 2508.0, 2454.0,    NAN,
        2904.0, 4354.0, 4698.0, 4600.0, 4644.0,    NAN,    NAN,
        3502.0, 5958.0, 6070.0, 6142.0,    NAN,    NAN,    NAN,
        2812.0, 4882.0, 4852.0,    NAN,    NAN,    NAN,    NAN,
        2642.0, 4406.0,    NAN,    NAN,    NAN,    NAN,    NAN,
        5022.0,    NAN,    NAN,    NAN,    NAN,    NAN,    NAN,
    ];
    PaidIncurredDataset {
        paid: Triangle::new(7, 7, paid),
        incurred: Triangle::new(7, 7, incurred),
    }
}

/// Friedland (2010), *Estimating Unpaid Claims Using Basic Techniques* (CAS),
/// US Industry Auto 10×10 Paid Claims, Reported Claims, and Earned Premium.
pub fn friedland_us_industry_auto() -> FriedlandAutoDataset {
    #[rustfmt::skip]
    let paid = vec![
        18539254.0, 33231039.0, 40062008.0, 43892039.0, 45896535.0, 46765422.0, 47221322.0, 47446877.0, 47555456.0, 47644187.0,
        20410193.0, 36090684.0, 43259402.0, 47159241.0, 49208532.0, 50162043.0, 50625757.0, 50878808.0, 51000534.0,        NAN,
        22120843.0, 38976014.0, 46389282.0, 50562385.0, 52735280.0, 53740101.0, 54284334.0, 54533225.0,        NAN,        NAN,
        22992259.0, 40096198.0, 47767835.0, 52093916.0, 54363436.0, 55378801.0, 55878421.0,        NAN,        NAN,        NAN,
        24092782.0, 41795313.0, 49903803.0, 54352884.0, 56754376.0, 57807215.0,        NAN,        NAN,        NAN,        NAN,
        24084451.0, 41399612.0, 49070332.0, 53584201.0, 55930654.0,        NAN,        NAN,        NAN,        NAN,        NAN,
        24369770.0, 41489863.0, 49236678.0, 53774672.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        25100697.0, 42702229.0, 50644994.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        25608776.0, 43606497.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        27229969.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
    ];
    #[rustfmt::skip]
    let reported = vec![
        37017487.0, 43169009.0, 45568919.0, 46784558.0, 47337318.0, 47533264.0, 47634419.0, 47689655.0, 47724678.0, 47742304.0,
        38954484.0, 46045718.0, 48882924.0, 50219672.0, 50729292.0, 50926779.0, 51069285.0, 51163540.0, 51185767.0,        NAN,
        41155776.0, 49371478.0, 52358476.0, 53780322.0, 54303086.0, 54582950.0, 54742188.0, 54837929.0,        NAN,        NAN,
        42394069.0, 50584112.0, 53704296.0, 55150118.0, 55895583.0, 56156727.0, 56299562.0,        NAN,        NAN,        NAN,
        44755243.0, 52971643.0, 56102312.0, 57703851.0, 58363564.0, 58592712.0,        NAN,        NAN,        NAN,        NAN,
        45163102.0, 52497731.0, 55468551.0, 57015411.0, 57565344.0,        NAN,        NAN,        NAN,        NAN,        NAN,
        45417309.0, 52640322.0, 55553673.0, 56976657.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        46360869.0, 53790061.0, 56786410.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        46582684.0, 54641339.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        48853563.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
    ];
    let earned_premium = vec![
        68574209.0, 68544981.0, 68907977.0, 72544955.0, 79228887.0, 86643542.0, 91763523.0,
        94115312.0, 95272279.0, 95176240.0,
    ];
    FriedlandAutoDataset {
        paid: Triangle::new(10, 10, paid),
        reported: Triangle::new(10, 10, reported),
        earned_premium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn revalidate(t: &Triangle) {
        let checked = Triangle::try_new(t.n_origins(), t.n_devs(), t.data().to_vec());
        assert_eq!(checked.as_ref(), Ok(t));
    }

    #[test]
    fn datasets_pass_try_new_validation() {
        revalidate(&raa());
        revalidate(&genins());
        let qm = quarg_mack_2004();
        revalidate(&qm.paid);
        revalidate(&qm.incurred);
        let fr = friedland_us_industry_auto();
        revalidate(&fr.paid);
        revalidate(&fr.reported);
        assert_eq!(fr.earned_premium.len(), fr.paid.n_origins());
    }
}
