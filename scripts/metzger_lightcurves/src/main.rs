//! Tabulate survey_sim's one-zone Metzger kilonova light curves.
//!
//! Calls `lightcurve_fitting::metzger_kn_mags` -- the function survey_sim's `ParametricModel::evaluate_metzger`
//! uses -- at the lightcurve-fitting revision pinned by survey_sim's Cargo.lock, with survey_sim's band wavelengths.
//! Evaluated at 10 pc and z = 0, so the output is the absolute AB magnitude without extinction.
//! Usage: cargo run --release > metzger_lightcurves.csv   (driven by scripts/make_metzger_check.py)

const C_CMS: f64 = 2.998e10; // as in survey_sim/src/lightcurve/parametric.rs
const PC_CM: f64 = 3.0857e18;

fn main() {
    // (label, M_ej [Msun], v_ej [c], kappa parameter) -- Suite 01 template and Suite 07 presets
    let models: [(&str, f64, f64, f64); 3] = [
        ("metzger_template_k398", 0.00126, 0.50, 398.0),
        ("metzger_blue_k0.5", 0.020, 0.25, 0.5),
        ("metzger_mean_k1", 0.040, 0.15, 1.0),
    ];
    let bands = [("g", 4770.0), ("r", 6231.0), ("i", 7625.0)]; // survey_sim band_wavelength_angstrom
    let freqs: Vec<(&str, f64)> = bands.iter().map(|(b, l)| (*b, C_CMS / (l * 1e-8))).collect();
    let times: Vec<f64> = (0..300).map(|k| 0.1 * 10f64.powf(k as f64 * 2.4 / 299.0)).collect(); // 0.1-25 d
    println!("param_set,inclination_deg,band,t_days,abs_mag");
    for (name, mej, vej, kappa) in models.iter() {
        let p = [mej.log10(), vej.log10(), kappa.log10(), 0.0];
        let mags = lightcurve_fitting::metzger_kn_mags(&p, &times, &freqs, 10.0 * PC_CM);
        for (b, _) in bands.iter() {
            for (t, m) in times.iter().zip(mags[*b].iter()) {
                println!("{},,{},{:.5},{:.4}", name, b, t, m);
            }
        }
    }
}
