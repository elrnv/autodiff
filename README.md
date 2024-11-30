# autofloat

![license](https://img.shields.io/badge/License-MIT-blue)
[![workflow](https://github.com/Rookfighter/autofloat/actions/workflows/rust.yml/badge.svg)](https://github.com/Rookfighter/autofloat/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/Rookfighter/autofloat/graph/badge.svg?token=DC8GWI6PLW)](https://codecov.io/gh/Rookfighter/autofloat)

`autofloat` is a pure Rust library, which implements efficient automatic differentiation in forward mode.

The library currently provides scalar datatypes to efficiently compute gradients.

Optionally you can use the library with the `nalgebra` feature to compute gradients and jacobians using the (nalgebra)[https://github.com/dimforge/nalgebra] library.

# Usage

# Examples

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
