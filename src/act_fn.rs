use ndarray::{Array1, Array2, Axis};

/// theta = 0.0
pub fn step(x: f64) -> f64 {
    let res = if x > 0.0 { 1.0 } else { 0.0 };
    res
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn array_relu(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&x| relu(x))
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let expsum: f64 = x.iter().map(|&x| x.exp()).sum();
    x.map(|&x| x.exp() / expsum)
}

pub fn batch_softmax(x: &Array2<f64>) -> Array2<f64> {
    let x_exp: Array2<f64> = x.map(|x| x.exp());
    let expsum: Array2<f64> = axis_sum(&x_exp, 1);
    let a = x_exp / expsum;
    a
}

/// axis = 0: sum of each row
/// axis = 1: sum of each column
pub fn axis_sum(x: &Array2<f64>, axis: usize) -> Array2<f64> {
    if axis != 0 && axis != 1 {
        panic!("axis must be 0 or 1");
    }

    let res: Array2<f64> = if axis == 0 {
        let mut sum: Array2<f64> = Array2::zeros((1, x.shape()[1]));
        for i in 0..x.shape()[0] {
            sum += &x.slice(s![i, ..]).insert_axis(Axis(axis));
        }
        sum
    } else {
        let mut sum: Array2<f64> = Array2::zeros((x.shape()[0], 1));
        for i in 0..x.shape()[1] {
            sum += &x.slice(s![.., i]).insert_axis(Axis(axis));
        }
        sum
    };

    res
}
