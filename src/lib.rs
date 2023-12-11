use std::rc::Rc;

// #[macro_use]
// extern crate ndarray;
use ndarray::{Array1, Array2};

pub mod act_fn;

// pub trait LayerBase {
//     fn weight_matrix(&self) -> &Array2<f64>;
//     fn bias_array(&self) -> &Array1<f64>;
//     fn activate(&self, x: Array1<f64>) -> Array1<f64>;
//     fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
//         self.activate(x.dot(self.weight_matrix()) + self.bias_array())
//     }
//     // fn backward(&self, x: &Array1<f64>, dx: &Array1<f64>) -> &Array1<f64>;
// }

pub enum LayerType {
    Hidden,
    Output,
}

pub struct Layer {
    pub weight_matrix: Array2<f64>,
    bias_array: Array1<f64>,
    pub output_array: Array2<f64>,
    pub du: Array2<f64>,
    layer_type: LayerType,
    next_layer: Option<Rc<Layer>>,
    prev_layer: Option<Rc<Layer>>,
}

impl Layer {
    pub fn new(input_synapse_num: usize, cell_num: usize, layer_type: LayerType, batch_size: usize) -> Layer {
        Layer {
            weight_matrix: Array2::zeros((input_synapse_num, cell_num)),
            bias_array: Array1::zeros(cell_num),
            output_array: Array2::zeros((batch_size, cell_num)),
            du: Array2::zeros((batch_size, cell_num)),
            layer_type,
            next_layer: None,
            prev_layer: None,
        }
    }



    /// 活性化関数：ソフトマックス関数 として実装
    pub fn calc_du_output_layer(&mut self, t: &Array2<f64>) {
        match self.layer_type {
            LayerType::Output => {
                if t.shape() != self.output_array.shape() {
                    panic!("t.shape() != self.output_array.shape()");
                }
                // dE/du を解析的に計算
                self.du = (-t) * (Array2::from_elem(self.output_array.dim(), 1.0) - &self.output_array);
            },
            _ => {
                panic!("calc_du_output_layer() is only for output layer");
            }
        }
    }

    /// 活性化関数：シグノイド関数 として実装
    pub fn calc_du_hidden_layer(&mut self) {
        match self.layer_type {
            LayerType::Hidden => {
                // dE/du を解析的に計算


                let next_layer_du = self.next_layer.as_ref().unwrap().du.clone();
                let next_layer_weight_matrix = self.next_layer.as_ref().unwrap().weight_matrix.clone();
                let next_layer_weight_matrix = next_layer_weight_matrix.t();

                let dy_du = &self.output_array * (Array2::from_elem(self.output_array.dim(), 1.0) - &self.output_array);
                let de_dy = next_layer_du.dot(&next_layer_weight_matrix);

                self.du = dy_du * de_dy;
            },
            _ => {
                panic!("calc_du_hidden_layer() is only for hidden layer");
            }
        }
    }
}

// impl LayerBase for Layer {
//     fn weight_matrix(&self) -> &Array2<f64> {
//         &self.weight_matrix
//     }
//     fn bias_array(&self) -> &Array1<f64> {
//         &self.bias_array
//     }
//     fn activate(&self, x: Array1<f64>) -> Array1<f64> {
//         (self.activate)(x)
//     }
// }

// pub struct Network {
//     layers: Vec<Layer>,
// }

// impl Network {
//     pub fn new(layers: Vec<Layer>) -> Network {
//         Network {
//             layers: layers,
//         }
//     }

//     pub fn output(&self, x: &Array1<f64>) -> Array1<f64> {
//         let mut output_array: Array1<f64> = x.clone();
//         for layer in &self.layers {
//             output_array = layer.forward(&output_array);
//         }
//         output_array
//     }
// }
