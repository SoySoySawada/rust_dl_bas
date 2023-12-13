use std::{rc::Rc, cell::RefCell};

use ndarray::{Array1, Array2};

use crate::act_fn;

pub enum LayerType {
    Hidden,
    Output,
}

pub struct Layer {
    pub weight_matrix: Array2<f64>,
    pub bias_array: Array1<f64>,
    pub output_array: Array2<f64>,
    pub du: Array2<f64>,
    pub dv_output: Array2<f64>,
    layer_type: LayerType,
    next_layer: Option<Rc<RefCell<Layer>>>,
    prev_layer: Option<Rc<RefCell<Layer>>>,
}

impl Layer {
    pub fn new(input_synapse_num: usize, cell_num: usize, layer_type: LayerType, batch_size: usize) -> Layer {
        Layer {
            weight_matrix: Array2::zeros((input_synapse_num, cell_num)),
            bias_array: Array1::from_elem(cell_num, 1.0),
            output_array: Array2::zeros((batch_size, cell_num)),
            du: Array2::zeros((batch_size, cell_num)),
            dv_output: Array2::zeros((batch_size, cell_num)),
            layer_type,
            prev_layer: None,
            next_layer: None,
        }
    }
    pub fn have_prev_layer(&self) -> bool {
        self.prev_layer.is_some()
    }
    pub fn have_next_layer(&self) -> bool {
        self.next_layer.is_some()
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
        // println!("output_array = {:?}", &self.output_array);
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

        // println!("output_array = {:?}", &self.output_array);
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

                println!("next_layer_du = {:?}", &next_layer_du);
                println!("dy_du = {:?}", &dy_du);   // DEL 出力は今回同じだから同じ値になるのは変じゃない
                println!("de_dy = {:?}", &de_dy);   // DEL 変じゃね b*n(l-1)になるはず？

                self.du = dy_du * de_dy;
            },
            _ => {
                panic!("calc_du_hidden_layer() is only for hidden layer");
            }
        }
    }

    pub fn update_first_layer(&mut self, x: &Array2<f64>, learn_rate: f64) {
        self.weight_matrix = &self.weight_matrix - learn_rate * self.calc_dw_first_layer(x);
        self.bias_array = &self.bias_array - learn_rate * self.calc_db();
    }

    fn calc_dw_first_layer(&self, x: &Array2<f64>) -> Array2<f64> {
        println!("x = {:?}", &x);
        let dw = x.t().dot(&self.du);
        println!("dw = {:?}", &dw);
        dw
    }

    pub fn update_other_layer(&mut self, learn_rate: f64) {
        self.weight_matrix = &self.weight_matrix - learn_rate * self.calc_dw_other_layer();
        self.bias_array = &self.bias_array - learn_rate * self.calc_db();
    }

    fn calc_dw_other_layer(&self) -> Array2<f64> {
        let prev_layer_output_array = self.prev_layer.as_ref().unwrap().borrow().output_array.clone();
        println!("prev_layer_output_array = {:?}", &prev_layer_output_array);
        let dw = prev_layer_output_array.t().dot(&self.du);
        println!("dw = {:?}", &dw);
        dw
    }

    fn calc_db(&self) -> Array1<f64> {
        let db = self.du.sum_axis(ndarray::Axis(0));
        db
    }

    fn calc_do_output_layer(&self, t: &Array2<f64>) -> Array2<f64> {
        let do_output: Array2<f64> = (-t) / &self.output_array;
        do_output
    }

    // fn calc_do_other_layer(&self) -> Array2<f64> {
    //     let next_layer_do: Array2<f64> = self.next_layer.as_ref().unwrap().borrow().dv_output.clone();
    //     let dy_dy: Array2<f64> = self.output_array.clone().map()
    //     let do_output = 
    // }
}

trait LayerBase {
    /// 出力層: その場で計算
    /// 隠れ層: プロパティに保持している値を返却
    fn div_output(&self) -> Array2<f64>;
}

/// ソフトマックス関数をするやつ
struct OutputLayer {
    output: Array2<f64>,
    output_div: Array2<f64>,
    input_div: Array2<f64>,
}

impl OutputLayer {
    fn calc_output_div(&mut self, t: &Array2<f64>) {
        let output_div: Array2<f64> = (-t) / &self.output;
        self.output_div = output_div
    }

    fn calc_input_div(&mut self) {
        let mut div_calc1: Array2<f64> = Array2::zeros(self.input_div.dim());

        for batch_idx in 0..self.input_div.shape()[0] {
            for input_idx in 0..self.input_div.shape()[1] {
                for output_idx in 0..self.output.shape()[1] {
                    div_calc1[[batch_idx, input_idx]] = if input_idx == output_idx {
                        self.output[[batch_idx, output_idx]] * (1.0 - self.output[[batch_idx, output_idx]])
                    } else {
                        -self.output[[batch_idx, output_idx]] * self.output[[batch_idx, input_idx]]
                    }
                }
            }
        }
    }
}