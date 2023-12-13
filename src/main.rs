#[macro_use]
extern crate ndarray;

use std::{rc::Rc, cell::RefCell};

use ndarray::{arr2, Array, Array1, Array2, Zip};

use rust_dl_bas::{act_fn, layer::{Layer, LayerType}, network::Network};




fn main() {
    let batch_size = 2;
    let input_size = 3;
    let output_size = 3;
    let hidden_size = 3;

    let mut network = Network::new();
    let hidden_layer1 = Rc::new(RefCell::new(Layer::new(input_size, hidden_size, LayerType::Hidden, batch_size)));
    // let hidden_layer2 = Rc::new(RefCell::new(Layer::new(hidden_size, hidden_size, LayerType::Hidden, batch_size)));
    let output_layer = Rc::new(RefCell::new(Layer::new(hidden_size, output_size, LayerType::Output, batch_size)));
    network.add_layer(hidden_layer1);
    // network.add_layer(hidden_layer2);
    network.add_layer(output_layer);

    println!("layer0 have prev:{}", network.layers[0].as_ref().borrow_mut().have_prev_layer());
    println!("layer0 have next:{}", network.layers[0].as_ref().borrow_mut().have_next_layer());
    println!("layer1 have prev:{}", network.layers[1].as_ref().borrow_mut().have_prev_layer());
    println!("layer1 have next:{}", network.layers[1].as_ref().borrow_mut().have_next_layer());
    // println!("layer2 have prev:{}", network.layers[2].as_ref().borrow_mut().have_prev_layer());
    // println!("layer2 have next:{}", network.layers[2].as_ref().borrow_mut().have_next_layer());
    

    let mut cnt = 0;
    while cnt < 1 {
        let testdata: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let answer: Array2<f64> = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        network.train(&testdata, &answer);
        println!("layer0 weight = {:?}", &network.layers[0].as_ref().borrow_mut().weight_matrix);
        println!("layer0 bias = {:?}", &network.layers[0].as_ref().borrow_mut().bias_array);
        cnt += 1;
    }

    println!("train end");
    

    network.calc_result(&array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let result = network.result();

    println!("result = {:?}", &result);
}

pub fn mat() {
    let a = arr2(&[[1, 2, 3],
        [4, 5, 6]]);

    println!("a = \n{:?}", &a);

    assert_eq!(a.get([0,1]), Some(&2));
    assert_eq!(a.get((0,3)), None);
    assert_eq!(a[(1, 2)], 6);
    assert_eq!(a[(1, 1)], 5);

    let mut index: i16 = 0;
    for e in a.iter() {
        index += 1;
        println!("e{} = {}", index, e);
    }

    let mut m: Array2<f64> = Array::zeros((10, 10));
    println!("m = \n{:?}", &m);
    let mut fill_val = 1.0;
    for mut row in m.rows_mut() {
        row.fill(fill_val);
        fill_val += 1.5;
    }
    println!("m = \n{:?}", &m);


    type M = Array2<f64>;
    let mut a: M = M::zeros((12, 8));
    let b = M::from_elem(a.dim(), 1.0);
    let c = M::from_elem(a.dim(), 2.0);
    let d = M::from_elem(a.dim(), 3.0);

    Zip::from(&mut a)
        .and(&b)
        .and(&c)
        .and(&d)
        .for_each(|w, &x, &y, &z| {
            *w = x + y * z ;
        });
    println!("{:?}", &a);

    let mut t = Array::zeros((5, 5, 5));
    let mut idx = 0;
    for elm in t.iter_mut() {
        *elm = idx;
        idx += 1;
    }
    println!("{:?}", t.slice(s![.., 0..2, 2..4]));
}

pub fn test_perceptron() {
    let x: Array1<f64> = array![1.0, 1.0];
    println!("x = {:?}", &x);
    println!("and = {}", and(&x));
    println!("or = {}", or(&x));
    println!("nand = {}", nand(&x));
    println!("xor = {}", xor(&x));
}

fn over(x: f64, theta: f64) -> f64 {
    return if x >= theta {1.0} else {0.0};
}

pub fn and(x: &Array1<f64>) -> f64 {
    let w: Array1<f64> = array![0.5, 0.5];
    let theta: f64 = 0.7;
    over(w.dot(x), theta)
}

pub fn or(x: &Array1<f64>) -> f64 {
    let w: Array1<f64> = array![0.5, 0.5];
    let theta: f64 = 0.3;
    over(w.dot(x), theta)
}

pub fn nand(x: &Array1<f64>) -> f64 {
    let w: Array1<f64> = array![-0.5, -0.5];
    let theta: f64 = -0.7;
    over(w.dot(x), theta)
}

pub fn xor(x: &Array1<f64>) -> f64 {
    let y1 = or(x);
    let y2 = nand(x);
    and(&array![y1, y2])
}

pub fn forward() {
    let x: Array1<f64> = array![1.0, 0.5];  // 入力値
    let mut weight_arrays: Vec<Array2<f64>> = Vec::new();
    let mut bias_arrays: Vec<Array1<f64>> = Vec::new();

    weight_arrays.push(array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
    weight_arrays.push(array![[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]);
    weight_arrays.push(array![[0.1, 0.3],[0.2, 0.4]]);

    bias_arrays.push(array![0.1, 0.2, 0.3]);
    bias_arrays.push(array![0.1, 0.2]);
    bias_arrays.push(array![0.1, 0.2]);

    let mut voting_array: Array1<f64>;
    let mut output_array: Array1<f64> = x;
    for i in 0..3 {
        voting_array = output_array.dot(&weight_arrays[i]) + &bias_arrays[i];

        if i < 2 {
            output_array = act_fn::array_relu(&voting_array);
        } else {
            output_array = act_fn::softmax(&voting_array);
        }
    }

    println!("output_array = /r/n{:?}", &output_array);
}