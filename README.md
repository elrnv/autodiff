# `autodiff`

An auto-differentiation library.

[![On crates.io](https://img.shields.io/crates/v/autodiff.svg)](https://crates.io/crates/autodiff)
[![On docs.rs](https://docs.rs/autodiff/badge.svg)](https://docs.rs/autodiff/)
[![Build status](https://travis-ci.org/elrnv/autodiff.svg?branch=master)](https://travis-ci.org/elrnv/autodiff)

Currently supported features:

  - [x] Forward auto-differentiation

  - [ ] Reverse auto-differentiation

To compute a derivative with respect to a variable using this library:

  1. create a variable of type `F`, which implements the `Float` trait from the `num-traits` crate.

  2. compute your function using this variable as the input.

  3. request the derivative from this variable using the `deriv` method.


# Disclaimer

This library is a work in progress and is not ready for production use.


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

# Features

Support for `approx`, `cgmath` and `nalgebra` via the `approx`, `cgmath` and `na` feature flags respectively.

# License

This repository is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.


# Acknowledgements

This library started as a fork of [rust-ad](https://github.com/ibab/rust-ad).
