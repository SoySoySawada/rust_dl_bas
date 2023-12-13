#[macro_use]
extern crate ndarray;

use std::{rc::Rc, cell::RefCell};

use ndarray::Array2;

use rust_dl_bas::{ layer::{HiddenLayer, OutputLayer, LayerBase}, network::Network};




fn main() {
    let batch_size = 3;
    let input_size = 2;
    let output_size = 3;
    let hidden_size = 8;

    let mut network = Network::new();
    let hidden_layer1 = Rc::new(RefCell::new(HiddenLayer::new(input_size, hidden_size, batch_size)));
    let output_layer = Rc::new(RefCell::new(OutputLayer::new_rand(hidden_size, output_size, batch_size)));

    println!("hidden weight = {:?}", &hidden_layer1.borrow().weight());
    println!("output weight = {:?}", &output_layer.borrow().weight());

    network.add_layer(hidden_layer1);
    network.add_layer(output_layer);
    

    let mut cnt = 0;
    while cnt < 1000 {
        let testdata: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let answer: Array2<f64> = array![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        network.train(&testdata, &answer);
        cnt += 1;
    }

    println!("train end");
    
    // println!("layer0 weight = {:?}", &network.layers[0].borrow().weight());

    network.calc_result(&array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]);
    let result = network.result();

    println!("result = {:?}", &result);
}
