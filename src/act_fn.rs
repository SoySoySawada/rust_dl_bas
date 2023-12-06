use ndarray::Array1;



/// theta = 0.0
pub fn step(x: f64) -> f64 {
    let res = if x > 0.0 {1.0} else {0.0};
    res
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn array_relu(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&x| relu(x))
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {x} else {0.0}
}

pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let expsum: f64 = x.iter().map(|&x| x.exp()).sum();
    x.map(|&x| x.exp() / expsum)
}