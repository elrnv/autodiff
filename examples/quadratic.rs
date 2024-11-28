use autofloat::*;

fn main() {
    let x: f64 = 0.0;
    let f = |x: AutoFloat1<f64>| -> AutoFloat1<f64> { (x - 1.0f64).pow(2.0) };

    let dfdx = f(AutoFloat1::variable(x, 0));

    println!("f(x) = (x-1)^2");
    println!("df/dx = {:?} at x = {}", dfdx.dx, x,);
}
