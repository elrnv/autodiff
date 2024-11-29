# autofloat

![license](https://img.shields.io/badge/License-MIT-blue)
[![workflow](https://github.com/Rookfighter/autofloat/actions/workflows/rust.yml/badge.svg)](https://github.com/Rookfighter/autofloat/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/Rookfighter/autofloat/graph/badge.svg?token=DC8GWI6PLW)](https://codecov.io/gh/Rookfighter/autofloat)

`autofloat` is a pure Rust library, which implements efficient automatic differentiation in forward mode.

The library currently provides scalar datatypes to compute efficiently compute gradients.
Optionally you can use the library with `nalgebra`support to compute gradients and jacobians.

# Install

# Usage

# Examples

The following example differentiates a 1D function defined by a closure.

```rust
    // Define a function `f(x) = e^{-0.5*x^2}`.
    let f = |x: FT<f64>| (-x * x / F1::cst(2.0)).exp();

    // Differentiate `f` at zero.
    println!("{}", diff(f, 0.0)); // prints `0`
```

To compute the gradient of a function, use the function `grad` as follows:

```rust
    // Define a function `f(x,y) = x*y^2`.
    let f = |x: &[FT<f64>]| x[0] * x[1] * x[1];

    // Differentiate `f` at `(1,2)`.
    let g = grad(f, &vec![1.0, 2.0]);
    println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
```


Compute a specific derivative of a multi-variable function:

```rust
     // Define a function `f(x,y) = x*y^2`.
     let f = |v: &[FT<f64>]| v[0] * v[1] * v[1];

     // Differentiate `f` at `(1,2)` with respect to `x` (the first unknown) only.
     let v = vec![
         F1::var(1.0), // Create a variable.
         F1::cst(2.0), // Create a constant.
     ];
     println!("{}", f(&v).deriv()); // prints `4`
```


# License

This repository is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

# Acknowledgements

This library started as a fork of [autodiff](https://github.com/elrnv/autodiff) by Egor Larionov, which again was forked from [rust-ad](https://github.com/ibab/rust-ad) by Igor Babuschkin. Some parts of the code base still contain signifcant parts of these upstream repositories.
