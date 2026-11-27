use std::io;

use reserving_kernels::Triangle;
use reserving_kernels::preprocessing::exhibit::build_exhibit_bundle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let data = vec![
        1000.0, 1500.0, 1700.0, 1800.0,
        1200.0, 1800.0, 2000.0, f64::NAN,
        1500.0, 2100.0, f64::NAN, f64::NAN,
        1800.0, f64::NAN, f64::NAN, f64::NAN,
    ];
    let triangle = Triangle::try_new(4, 4, data)?;
    let premium = [4000.0, 4500.0, 5000.0, 5500.0];
    let expected_ultimate = [2600.0, 2800.0, 3200.0, 3600.0];
    let bundle = build_exhibit_bundle(
        &triangle.view(),
        Some(&premium),
        Some(&expected_ultimate),
        1.0,
    )?;

    serde_json::to_writer_pretty(io::stdout(), &bundle)?;
    println!();
    Ok(())
}
