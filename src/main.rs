#[macro_use]
extern crate ndarray;

use ndarray::Array2;

use rust_dl_bas::basic_neural::Neural;




fn main() {
    main_neural();
    // let batch_size = 3;
    // let input_size = 2;
    // let output_size = 3;
    // let hidden_size = 8;

    // let mut network = Network::new();
    // let hidden_layer1 = Rc::new(RefCell::new(HiddenLayer::new(input_size, hidden_size, batch_size)));
    // let output_layer = Rc::new(RefCell::new(OutputLayer::new_rand(hidden_size, output_size, batch_size)));

    // println!("hidden weight = {:?}", &hidden_layer1.borrow().weight());
    // println!("output weight = {:?}", &output_layer.borrow().weight());

    // network.add_layer(hidden_layer1);
    // network.add_layer(output_layer);
    

    // let mut cnt = 0;
    // while cnt < 1000 {
    //     let testdata: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
    //     let answer: Array2<f64> = array![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    //     network.train(&testdata, &answer);
    //     cnt += 1;
    // }

    // println!("train end");
    
    // // println!("layer0 weight = {:?}", &network.layers[0].borrow().weight());

    // network.calc_result(&array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]);
    // let result = network.result();

    // println!("result = {:?}", &result);
}

fn main_neural() {
    let x: Array2<f64> = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let t: Array2<f64> = array![[1., 0.], [0., 1.], [0., 1.], [1., 0.]];
    let n = x.shape()[0];

    let input_size = x.shape()[1];
    let hidden_size = 2;
    let output_size = 2;
    let epsilon = 0.1;
    let mu = 0.9;
    let epoch = 10000;

    let mut nn = Neural::new(input_size, hidden_size, output_size);
    let input = x.clone();
    nn.init();
    nn.train(input, t, epsilon, mu, epoch);
    let _error = nn.error();

    let input = x.clone();
    let y = nn.predict(input);

    for i in 0..n {
        println!("{}: {} -> {}", i, x.slice(s![i, ..]), y.slice(s![i, ..]));
    }    
}