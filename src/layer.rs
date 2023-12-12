use std::{rc::Rc, cell::RefCell};

use ndarray::{Array1, Array2};

use crate::act_fn;

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
    next_layer: Option<Rc<RefCell<Layer>>>,
    prev_layer: Option<Rc<RefCell<Layer>>>,
}

impl Layer {
    pub fn new(input_synapse_num: usize, cell_num: usize, layer_type: LayerType, batch_size: usize) -> Layer {
        Layer {
            weight_matrix: Array2::zeros((input_synapse_num, cell_num)),
            bias_array: Array1::zeros(cell_num),
            output_array: Array2::zeros((batch_size, cell_num)),
            du: Array2::zeros((batch_size, cell_num)),
            layer_type,
            prev_layer: None,
            next_layer: None,
        }
    }
    
    pub fn set_next_layer(&mut self, next_layer: Rc<RefCell<Layer>>) {
        self.next_layer.replace(next_layer);
    }

    pub fn set_prev_layer(&mut self, prev_layer: Rc<RefCell<Layer>>) {
        self.prev_layer.replace(prev_layer);
    }

    // hidden層 活性化関数：シグノイド関数 として実装
    pub fn calc_output_first_layer(&mut self, x: &Array2<f64>) {
        if self.prev_layer.is_some() {
            panic!("is not first layer");
        }

        let u = x.dot(&self.weight_matrix) + &self.bias_array;

        self.output_array = u.map(|u| act_fn::sigmoid(*u));
    }

    pub fn calc_output_other_layer(&mut self) {
        if self.prev_layer.is_none() {
            panic!("is first layer");
        }

        let prev_layer_output_array = self.prev_layer.as_ref().unwrap().borrow().output_array.clone();

        let u = prev_layer_output_array.dot(&self.weight_matrix) + &self.bias_array;

        match self.layer_type {
            LayerType::Hidden => {
                self.output_array = u.map(|u| act_fn::sigmoid(*u));
            },
            LayerType::Output => {
                self.output_array = act_fn::batch_softmax(&u);
            },
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

                let next_layer_du = self.next_layer.as_ref().unwrap().borrow().du.clone();
                let next_layer_weight_matrix = self.next_layer.as_ref().unwrap().borrow().weight_matrix.clone();
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

    pub fn aa(&mut self) {
        let mut a = Layer::new(2, 3, LayerType::Hidden, 1);
        let mut a_2 = Layer::new(2, 3, LayerType::Hidden, 1);
        let b = Rc::new(RefCell::new(a));

        b.borrow_mut().set_next_layer(Rc::new(RefCell::new(a_2)));

        self.prev_layer = Some(Rc::clone(&b));
    }
}
