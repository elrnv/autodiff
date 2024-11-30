macro_rules! to_degrees_impl {
    ($v:expr) => {
        AutoFloat {
            x: $v.x.to_degrees(),
            dx: unary_op($v.dx, |v| v.to_degrees()),
        }
    };
}
pub(crate) use to_degrees_impl;

macro_rules! to_radians_impl {
    ($v:expr) => {
        AutoFloat {
            x: $v.x.to_radians(),
            dx: unary_op($v.dx, |v| v.to_radians()),
        }
    };
}
pub(crate) use to_radians_impl;

macro_rules! max_impl {
    ($lhs:expr, $rhs:expr) => {
        if $lhs.x < $rhs.x {
            $rhs
        } else {
            $lhs
        }
    };
}
pub(crate) use max_impl;

macro_rules! min_impl {
    ($lhs:expr, $rhs:expr) => {
        if $lhs.x > $rhs.x {
            $rhs
        } else {
            $lhs
        }
    };
}
pub(crate) use min_impl;

#[cfg(feature = "float_impl")]
mod dependent {
    macro_rules! floor_impl {
        ($lhs:expr) => {
            AutoFloat {
                x: $lhs.x.floor(),
                dx: $lhs.dx,
            }
        };
    }
    pub(crate) use floor_impl;

    macro_rules! ceil_impl {
        ($lhs:expr) => {
            AutoFloat {
                x: $lhs.x.ceil(),
                dx: $lhs.dx,
            }
        };
    }
    pub(crate) use ceil_impl;

    macro_rules! round_impl {
        ($lhs:expr) => {
            AutoFloat {
                x: $lhs.x.round(),
                dx: $lhs.dx,
            }
        };
    }
    pub(crate) use round_impl;

    macro_rules! trunc_impl {
        ($lhs:expr) => {
            AutoFloat {
                x: $lhs.x.trunc(),
                dx: $lhs.dx,
            }
        };
    }
    pub(crate) use trunc_impl;

    macro_rules! fract_impl {
        ($lhs:expr) => {
            AutoFloat {
                x: $lhs.x.fract(),
                dx: $lhs.dx,
            }
        };
    }
    pub(crate) use fract_impl;

    macro_rules! abs_impl {
        ($lhs:expr) => {
            if $lhs.x >= T::zero() {
                $lhs
            } else {
                -$lhs
            }
        };
    }
    pub(crate) use abs_impl;

    macro_rules! mul_add_impl {
        ($a:expr, $b:expr,$c:expr) => {{
            AutoFloat {
                x: $a.x.mul_add($b.x, $c.x),
                dx: binary_op(
                    binary_op($a.dx, $b.dx, |l, r| l * $b.x + r * $a.x),
                    $c.dx,
                    |l, r| l + r,
                ),
            }
        }};
    }
    pub(crate) use mul_add_impl;

    macro_rules! recip_impl {
        ($lhs:expr) => {{
            let factor = -T::one() / $lhs.x * $lhs.x;
            AutoFloat {
                x: $lhs.x.recip(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use recip_impl;

    macro_rules! sin_impl {
        ($lhs:expr) => {{
            let cos_x = $lhs.x.cos();
            AutoFloat {
                x: $lhs.x.sin(),
                dx: unary_op($lhs.dx, |v| v * cos_x),
            }
        }};
    }
    pub(crate) use sin_impl;

    macro_rules! cos_impl {
        ($lhs:expr) => {{
            let sin_x = -$lhs.x.sin();
            AutoFloat {
                x: $lhs.x.cos(),
                dx: unary_op($lhs.dx, |v| v * sin_x),
            }
        }};
    }
    pub(crate) use cos_impl;

    macro_rules! tan_impl {
        ($lhs:expr) => {{
            let tan_x = $lhs.x.tan();
            let factor = tan_x * tan_x + T::one();
            AutoFloat {
                x: tan_x,
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use tan_impl;

    macro_rules! asin_impl {
        ($lhs:expr) => {{
            let factor = T::one() / (T::one() - $lhs.x * $lhs.x).sqrt();
            AutoFloat {
                x: $lhs.x.asin(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use asin_impl;

    macro_rules! acos_impl {
        ($lhs:expr) => {{
            let factor = -T::one() / (T::one() - $lhs.x * $lhs.x).sqrt();
            AutoFloat {
                x: $lhs.x.acos(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use acos_impl;

    macro_rules! atan_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.x * $lhs.x + T::one());
            AutoFloat {
                x: $lhs.x.atan(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use atan_impl;

    macro_rules! atan2_impl {
        ($lhs:expr, $rhs:expr) => {{
            let factor = T::one() / ($lhs.x * $lhs.x + $rhs.x * $rhs.x);
            AutoFloat {
                x: $lhs.x.atan2($rhs.x),
                dx: binary_op($lhs.dx, $rhs.dx, |l, r| (l * $rhs.x - r * $lhs.x) * factor),
            }
        }};
    }
    pub(crate) use atan2_impl;

    macro_rules! sin_cos_impl {
        ($lhs:expr) => {{
            let (s, c) = $lhs.x.sin_cos();
            let sn = AutoFloat {
                x: s,
                dx: unary_op($lhs.dx, |v| v * c),
            };
            let s_neg = -s;
            let cn = AutoFloat {
                x: c,
                dx: unary_op($lhs.dx, |v| v * s_neg),
            };
            (sn, cn)
        }};
    }
    pub(crate) use sin_cos_impl;

    macro_rules! sinh_impl {
        ($lhs:expr) => {{
            let cosh_x = $lhs.x.cosh();
            AutoFloat {
                x: $lhs.x.sinh(),
                dx: unary_op($lhs.dx, |v| v * cosh_x),
            }
        }};
    }
    pub(crate) use sinh_impl;

    macro_rules! cosh_impl {
        ($lhs:expr) => {{
            let sinh_x = $lhs.x.sinh();
            AutoFloat {
                x: $lhs.x.cosh(),
                dx: unary_op($lhs.dx, |v| v * sinh_x),
            }
        }};
    }
    pub(crate) use cosh_impl;

    macro_rules! tanh_impl {
        ($lhs:expr) => {{
            let tanhx = $lhs.x.tanh();
            let factor = T::one() - tanhx * tanhx;
            AutoFloat {
                x: tanhx,
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use tanh_impl;

    macro_rules! asinh_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.x * $lhs.x + T::one()).sqrt();
            AutoFloat {
                x: $lhs.x.asinh(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use asinh_impl;

    macro_rules! acosh_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.x * $lhs.x - T::one()).sqrt();
            AutoFloat {
                x: $lhs.x.acosh(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use acosh_impl;

    macro_rules! atanh_impl {
        ($lhs:expr) => {{
            let factor = T::one() / (-$lhs.x * $lhs.x + T::one());
            AutoFloat {
                x: $lhs.x.atanh(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use atanh_impl;

    macro_rules! log_impl {
        ($lhs:expr, $rhs:expr) => {{
            let ln_bx = $rhs.x.ln();
            let factor_bdx = -$lhs.x.ln() / ($rhs.x * ln_bx * ln_bx);
            let factor_sdx = T::one() / $lhs.x * ln_bx;

            AutoFloat {
                x: $lhs.x.log($rhs.x),
                dx: binary_op($lhs.dx, $rhs.dx, |l, r| factor_bdx * r + factor_sdx * l),
            }
        }};
    }
    pub(crate) use log_impl;

    macro_rules! log2_impl {
        ($lhs:expr, $two:expr) => {{
            $lhs.log(AutoFloat::constant($two))
        }};
    }
    pub(crate) use log2_impl;

    macro_rules! log10_impl {
        ($lhs:expr, $ten:expr) => {{
            $lhs.log(AutoFloat::constant($ten))
        }};
    }
    pub(crate) use log10_impl;

    macro_rules! ln_impl {
        ($lhs:expr) => {{
            let factor = $lhs.x.recip();
            AutoFloat {
                x: $lhs.x.ln(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use ln_impl;

    macro_rules! ln_1p_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.x + T::one());
            AutoFloat {
                x: $lhs.x.ln_1p(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use ln_1p_impl;

    macro_rules! sqrt_impl {
        ($lhs:expr) => {{
            let denom = $lhs.x.sqrt() * T::from(2).unwrap();
            let factor = if denom.is_zero() {
                T::zero()
            } else {
                T::one() / denom
            };

            AutoFloat {
                x: $lhs.x.sqrt(),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use sqrt_impl;

    macro_rules! cbrt_impl {
        ($lhs:expr) => {{
            let x_cbrt = $lhs.x.cbrt();
            let denom = x_cbrt * x_cbrt * T::from(3).unwrap();
            let factor = if denom.is_zero() {
                T::zero()
            } else {
                T::one() / denom
            };

            AutoFloat {
                x: x_cbrt,
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use cbrt_impl;

    macro_rules! exp_impl {
        ($lhs:expr) => {{
            let exp = $lhs.x.exp();
            AutoFloat {
                x: exp,
                dx: unary_op($lhs.dx, |v| v * exp),
            }
        }};
    }
    pub(crate) use exp_impl;

    macro_rules! exp2_impl {
        ($lhs:expr) => {{
            let exp2 = $lhs.x.exp2();
            let factor = T::from(2).unwrap().ln() * exp2;
            AutoFloat {
                x: exp2,
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use exp2_impl;

    macro_rules! powi_impl {
        ($lhs:expr, $rhs:expr) => {{
            let factor = $lhs.x.powi($rhs - 1) * T::from($rhs).unwrap();
            AutoFloat {
                x: $lhs.x.powi($rhs),
                dx: unary_op($lhs.dx, |v| v * factor),
            }
        }};
    }
    pub(crate) use powi_impl;

    macro_rules! powf_impl {
        ($lhs:expr, $rhs:expr, $x:expr) => {{
            // Avoid division by zero.
            let factor = if $lhs.x.is_zero() && $x.is_zero() {
                T::zero()
            } else {
                $x * $rhs.x / $lhs.x
            };

            AutoFloat {
                x: $x,
                dx: binary_op($lhs.dx, $rhs.dx, |l, r| {
                    // Avoid imaginary values in the ln
                    let dn = if r.is_zero() {
                        T::zero()
                    } else {
                        $lhs.x.ln() * r
                    };

                    dn * $x + l * factor
                }),
            }
        }};
    }
    pub(crate) use powf_impl;

    macro_rules! exp_m1_impl {
        ($lhs:expr) => {{
            let exp_x = $lhs.x.exp();
            AutoFloat {
                x: $lhs.x.exp_m1(),
                dx: unary_op($lhs.dx, |v| v * exp_x),
            }
        }};
    }
    pub(crate) use exp_m1_impl;
}

#[cfg(feature = "float_impl")]
pub(crate) use dependent::*;
