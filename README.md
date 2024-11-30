# autofloat

![license](https://img.shields.io/badge/License-MIT-blue)
[![workflow](https://github.com/Rookfighter/autofloat/actions/workflows/rust.yml/badge.svg)](https://github.com/Rookfighter/autofloat/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/Rookfighter/autofloat/graph/badge.svg?token=DC8GWI6PLW)](https://codecov.io/gh/Rookfighter/autofloat)

`autofloat` is a pure Rust library, which implements efficient automatic differentiation in forward mode.

The library currently provides scalar datatypes to efficiently compute gradients.

Optionally you can use the library with the `nalgebra` feature to compute gradients and jacobians using the [nalgebra](https://github.com/dimforge/nalgebra) library.

# Usage

`autofloat`can compute derivatives for single and multivariate functions.
The library provides a float-like type `AutoFloat` to automatically compute the derivate while the target function is computed.

First, make sure that the function for which you want to compute a derivate can handle the `AutoFloat` type (either by generics or explicitly).
Then simply instantiate the variables for which you want to compute the derivative and pass them into your target function, that's it!

Here's a simple example, which computes the gradient of a function wrt. two input variables.
The function is implemented using generics and can be used with different floating point types.

```rust
use autofloat::AutoFloat2;
use num_traits::float::FloatCore;

// Define some target function for which we want to compute the derivative.
// This variant is generic in T, but you could also use the `AutoFloat` type directly.
fn quadratic_func<T>(x: T, y: T) -> T
where
    T: FloatCore,
{
    (x - T::one()) * (T::from(2).unwrap() * y - T::one())
}

fn main() {
    // Use AutoFloat2 because we use a 2-dimensional function and
    // we want a 2-dimensional gradient.
    // The first parameter determines the value of the variable.
    // The second prameter determines the index of the derivative
    // for this variable within the gradient vector.
    let x = AutoFloat2::variable(2.25, 0);
    let y = AutoFloat2::variable(-1.75, 1);

    let result = quadratic_func(x, y);

    println!(
        "result={} gradient_x={} gradient_y={}",
        result.x, result.dx[0], result.dx[1]
    );
}
```

See also the `examples/`directory for more examples on the different options how to define target functions.

# License

This repository is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

# Contributing

The easiest way to contribute is to log an issue for bugs or new features. I will then review and discuss the contents of ticket with you and eventually maybe implement it.

A faster way to get your features or fixes into `autofloat` is to file a pull request. There are just a few rules you should adhere:

* Provide a meaningful PR description
* All CI checks of the PR must succeed, otherwise it will not be merged
* Have a code coverage of at least 80% of the lines you want to contribute

I will review the PR and we will discuss how and if your PR can be merged.
If your PR might get large, feel free to log a ticket first and we can discuss the details before you implement everything.

# Acknowledgements

This library started as a fork of [autodiff](https://github.com/elrnv/autodiff) by Egor Larionov, which again was forked from [rust-ad](https://github.com/ibab/rust-ad) by Igor Babuschkin. Some parts of the code base still contain signifcant parts of these upstream repositories.
