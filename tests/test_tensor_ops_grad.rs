extern crate autograd as ag;
extern crate ndarray;

#[test]
fn get() {
    let mut env = ag::VariableEnvironment::new();
    let v = &env.slot().set(ndarray::arr1(&[1., 2., 3.]));

    env.run(|graph| {
        let var = graph.variable_map_by_id(graph.env());
        let v = var[v];
        let a: ag::Tensor<f64> = 2. * v;
        let z = a.access_elem(1);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn add_n() {
    let mut ctx = ag::VariableEnvironment::new();
    let v1 = ctx.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v2 = ctx.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v3 = ctx.slot().set(ndarray::arr1(&[1., 2., 3.]));
    ctx.run(|graph| {
        let v1 = graph.variable_by_id(v1);
        let v2 = graph.variable_by_id(v2);
        let v3 = graph.variable_by_id(v3);
        let z = graph.add_n(&[v1, v2, v3]);
        let g = graph.grad(&[z], &[v2]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v2], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn clip() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.clip(v, 1.5, 2.5);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn asinh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.asinh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn acosh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.acosh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn atanh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.atanh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn sinh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.sinh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn cosh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.cosh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn tanh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.tanh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn asin() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.asin(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn acos() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.acos(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn atan() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.atan(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn sin() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.sin(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn cos() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.cos(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn tan() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.tan(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn pow() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0.9, 1.1));
    env.run(|ctx| {
        let v = ctx.variable_by_id(v);
        let z = ctx.pow(v, 1.1);
        let g = ctx.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, ctx);
    });
}

#[test]
fn sqrt() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0.9, 1.1));
    env.run(|graph| {
        let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
        let v = graph.variable_by_id(v);
        let z = graph.sqrt(v);
        graph.add(v, z);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn exp() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0.9, 1.1));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.exp(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn ln() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 1., 1.1));
    env.run(|graph| {
        use std::f64;
        let v = graph.variable_by_id(v);
        let z = graph.ln(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn expand_dims() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.expand_dims(v, &[0, 2]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn squeeze() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 1, 2, 1]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.squeeze(v, &[3, 1]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn g_matmul() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let a = graph.convert_to_tensor(rng.standard_normal(&[4, 2]));
        let v = graph.variable_by_id(v);
        let z = graph.matmul(a, v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 5e-3, graph);
    });
}

#[test]
fn batch_matmul() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2, 3]));
    env.run(|graph| {
        let a = graph.convert_to_tensor(rng.standard_normal(&[2, 4, 2]));
        let v = graph.variable_by_id(v);
        let z = graph.batch_matmul(a, v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn implicit_broadcast() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let b = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let x = graph.convert_to_tensor(rng.standard_normal(&[4, 3]));
        let b = graph.variable_by_id(b);
        let z = x + b;
        let g = graph.grad(&[z], &[b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[b], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn wx_plus_b() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let w = env.slot().set(rng.standard_normal(&[2, 3]));
    let b = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let x = graph.convert_to_tensor(rng.standard_normal(&[4, 2]));
        let w = graph.variable_by_id(w);
        let b = graph.variable_by_id(b);
        let z = graph.matmul(x, w) + b;
        let g = graph.grad(&[z], &[b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[b], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_min() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_min(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_min_keep() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_min(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_max() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_max(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_max_keep() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_max(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_mean() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_mean(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_mean_keep() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_mean(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_sum() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_sum(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_sum_keep() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_sum(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reduce_prod() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_prod(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn maximum() {
    let mut env = ag::VariableEnvironment::new();
    let v1 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v2 = env.slot().set(ndarray::arr1(&[4., 5., 6.]));
    env.run(|graph| {
        let v1 = graph.variable_by_id(v1);
        let v2 = graph.variable_by_id(v2);
        let z = graph.maximum(v1, v2);
        let g = graph.grad(&[z], &[v1, v2]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v1, v2],
            &[],
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn minimum() {
    let mut env = ag::VariableEnvironment::new();
    let v1 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v2 = env.slot().set(ndarray::arr1(&[4., 5., 6.]));
    env.run(|graph| {
        let v1 = graph.variable_by_id(v1);
        let v2 = graph.variable_by_id(v2);
        let z = graph.minimum(v1, v2);
        let g = graph.grad(&[z], &[v1, v2]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v1, v2],
            &[],
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn abs() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.abs(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn neg() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.neg(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn square() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.square(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reciprocal() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[2, 3], 1., 1.01));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.inv(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn transpose() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 2, 3, 4]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.transpose(v, &[2, 3, 0, 1]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reshape_after_transpose() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3, 4]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.transpose(v, &[2, 1, 0]);
        let z = graph.reshape(z, &[4, 6]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn transpose_then_reshape_then_mm() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 2, 3, 4, 5]));
    let v2 = env.slot().set(rng.standard_normal(&[8, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let v2 = graph.variable_by_id(v2);
        let z = graph.transpose(v, &[4, 2, 3, 0, 1]);
        let z = graph.reshape(z, &[15, 8]);
        let z = graph.matmul(z, v2);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn add() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = env.slot().set(rng.standard_normal(&[2, 2]));
    let b = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let a = graph.variable_by_id(a);
        let b = graph.variable_by_id(b);
        let z = a + b;
        let g = graph.grad(&[z], &[a, b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[a], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn mul() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = env.slot().set(rng.standard_normal(&[2, 2]));
    let b = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let a = graph.variable_by_id(a);
        let b = graph.variable_by_id(b);
        let z = a * b;
        let g = graph.grad(&[z], &[a, b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[a], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn sigmoid() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.sigmoid(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn elu() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.elu(v, 1.);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn relu() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ag::ndarray::arr1(&[0.2, 0.5]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.relu(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn softplus() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.softplus(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn logsumexp() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reduce_logsumexp(v, 1, true);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn log_softmax() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.log_softmax(v, 1);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn softmax_cross_entropy() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let t = graph.convert_to_tensor(ndarray::arr2(&[[1., 0., 0.]]));
        let v = graph.variable_by_id(v);
        let z = graph.softmax_cross_entropy(v, t);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn sigmoid_cross_entropy() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let t = graph.convert_to_tensor(rng.standard_normal(&[1, 3]));
        let v = graph.variable_by_id(v);
        let z = graph.sigmoid_cross_entropy(v, t);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn sparse_softmax_cross_entropy() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let t = graph.convert_to_tensor(ndarray::arr1(&[1., 0.]));
        let v = graph.variable_by_id(v);
        let z = graph.sparse_softmax_cross_entropy(v, t);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn gather() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[5, 4, 8, 2]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let x = graph.convert_to_tensor(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));
        let z = graph.gather(v, x, 2);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn concat() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v1 = env.slot().set(rng.standard_normal(&[1, 2]));
    let v2 = env.slot().set(rng.standard_normal(&[1, 2]));
    env.run(|graph| {
        let v1 = graph.variable_by_id(v1);
        let v2 = graph.variable_by_id(v2);
        let z = graph.concat(&[v1, v2], 1);
        let g = graph.grad(&[z], &[v1]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v1, v2],
            &[],
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn slice() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.slice(v, &[0, 0], &[-1, 2]); // numpy equivalent is v[:, 0:2]
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn split() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 7, 5]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.split(v, &[2, 3, 2], 1);
        let g = graph.grad(&[&z[1]], &[v]);
        ag::test_helper::check_theoretical_grads(z[1], g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn flatten() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.flatten(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn reshape() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reshape(v, &[4, 2, 2]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
#[should_panic]
fn reshape_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable_by_id(v);
        let z = graph.reshape(&(v), &[4, 2, 2]);
        let g = graph.grad(&[z], &[v])[0];
        let gg = graph.grad(&[g], &[v]);
        ag::test_helper::check_theoretical_grads(g, gg.as_slice(), &[v], &[], 1e-3, 1e-3, graph);
    });
}

#[test]
fn conv2d_transpose() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[3, 2, 2, 2]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let w = graph.variable_by_id(w);
        let y = graph.conv2d_transpose(x, w, 0, 1);
        let g = graph.grad(&[y], &[w]);
        ag::test_helper::check_theoretical_grads(y, &g, &[w], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
#[should_panic]
fn conv2d_transpose_filter_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 2, 2, 2]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let w = graph.variable_by_id(w);
        let y = graph.conv2d_transpose(x, w, 0, 1);
        let g = graph.grad(&[y], &[w])[0];
        let gg = graph.grad(&[g], &[w]);
        ag::test_helper::check_theoretical_grads(g, &gg, &[w], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
#[should_panic]
fn conv2d_filter_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let w = graph.variable_by_id(w);
        let y = graph.conv2d(x, w, 0, 1);
        let g = graph.grad(&[y], &[w])[0];
        let gg = graph.grad(&[g], &[w]);
        ag::test_helper::check_theoretical_grads(g, &gg, &[w], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn conv2d_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    let gy = env.slot().set(ag::ndarray_ext::ones(&[2, 2, 2, 2]));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let w = graph.variable_by_id(w);
        let y = graph.conv2d(x, w, 0, 1);
        let gy = graph.variable_by_id(gy);
        unsafe {
            let g = graph.grad_with_default(&[y], &[x], &[gy])[0];
            let gg = graph.grad(&[g], &[gy])[0];
            ag::test_helper::check_theoretical_grads(g, &[gg], &[gy], &[], 1e-3, 1e-2, graph);
        }
    });
}

#[test]
fn conv2d_xw_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let w = graph.variable_by_id(w);
        let y = graph.conv2d(x, w, 0, 1);
        let g = graph.grad(&[y], &[w])[0];
        let gg = graph.grad(&[g], &[x]);
        ag::test_helper::check_theoretical_grads(g, &gg, &[x], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
#[should_panic]
fn conv2d_x_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let w = graph.variable_by_id(w);
        let y = graph.conv2d(x, w, 0, 1);
        let g = graph.grad(&[y], &[x])[0];
        let gg = graph.grad(&[g], &[x]); // can't differentiate with x twice
        ag::test_helper::check_theoretical_grads(y, &gg, &[x], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn conv2d() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 3, 3]));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let w = graph.variable_by_id(w);
        let y = graph.conv2d(x, w, 1, 2);
        let g = graph.grad(&[y], &[x, w]);
        ag::test_helper::check_theoretical_grads(y, &g, &[x, w], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn max_pool2d() {
    let mut env = ag::VariableEnvironment::new();
    let x = env.slot().set(ndarray::Array::linspace(0., 1., 9));
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let y = graph.max_pool2d(graph.reshape(x, &[1, 1, 3, 3]), 2, 0, 1);
        let g = graph.grad(&[y], &[x]);
        ag::test_helper::check_theoretical_grads(y, &g, &[x], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn max_pool2d_grad() {
    let mut env = ag::VariableEnvironment::new();
    let x = env.slot().set(ndarray::Array::linspace(0., 1., 36));
    let gy = env.slot().set(
        ndarray::Array::linspace(0., 1., 16)
            .into_shape(ndarray::IxDyn(&[2, 2, 2, 2]))
            .unwrap(),
    );
    env.run(|graph| {
        let x = graph.variable_by_id(x);
        let y = graph.max_pool2d(graph.reshape(x, &[2, 2, 3, 3]), 2, 0, 1);
        let gy = graph.variable_by_id(gy);
        unsafe {
            let g = graph.grad_with_default(&[y], &[x], &[gy])[0];
            let gg = graph.grad(&[g], &[gy])[0];
            ag::test_helper::check_theoretical_grads(g, &[gg], &[gy], &[], 1e-3, 1e-2, graph);
        }
    });
}

#[test]
fn tensordot() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = env.slot().set(rng.standard_normal(&[3, 4, 5]));
    env.run(|graph| {
        let a = graph.variable_by_id(a);
        let b = graph.convert_to_tensor(rng.standard_normal(&[4, 3, 2]));
        let c = graph.tensordot(a, b, &[1, 0], &[0, 1]);
        let g = graph.grad(&[c], &[a]);
        ag::test_helper::check_theoretical_grads(c, &g, &[a], &[], 1e-3, 1e-2, graph);
    });
}

#[test]
fn primitive_back_propagation_through_time() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let lookup_table = env.slot().set(rng.standard_normal(&[5, 3]));
    // (vector_dim -> vocab)
    let wo = env.slot().set(rng.standard_normal(&[3, 5]));
    // (vector_dim -> vector_dim)
    let wh = env.slot().set(rng.standard_normal(&[3, 3]));

    env.run(|graph| {
        let max_sent = 3;
        let batch_size = 2;

        let lookup_table = graph.variable_by_id(lookup_table);
        // (vector_dim -> vocab)
        let wo = graph.variable_by_id(wo);
        // (vector_dim -> vector_dim)
        let wh = graph.variable_by_id(wh);

        // -- build graph for BPTT --
        let mut loss_buf = vec![];
        let mut h_buf = vec![graph.placeholder(&[-1, max_sent])];
        let sentences = graph.placeholder(&[-1, max_sent]);

        for i in 0..max_sent {
            // pick new word id
            let id = graph.squeeze(graph.slice(sentences, &[0, i], &[-1, i + 1]), &[-1]);

            let new_h = {
                // recall last h
                let last_h = h_buf.last().unwrap();
                // compute and accumulate `loss`
                loss_buf.push(graph.sparse_softmax_cross_entropy(&graph.matmul(last_h, wo), &id));
                // new `h`
                graph.tanh(&(graph.gather(&lookup_table, &id, 0) + graph.matmul(last_h, wh)))
            };

            h_buf.push(new_h);
        }
        // last loss (after processed whole sequence)
        let loss = *loss_buf.last().unwrap();

        // inputs (batch_size=2, sentence_len=4)
        let params = &[lookup_table, wo, wh];
        let g = graph.grad(&[loss], params);
        ag::test_helper::check_theoretical_grads(
            loss,
            g.as_slice(),
            params,
            &[
                sentences.given(
                    ndarray::arr2(&[[2., 3., 1.], [0., 2., 0.]])
                        .into_dyn()
                        .view(),
                ),
                h_buf[0].given(rng.standard_normal(&[batch_size, 3]).view()),
            ],
            1e-3,
            1e-3,
            graph,
        );
    });
}
