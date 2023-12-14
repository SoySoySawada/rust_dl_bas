use std::{rc::Rc, cell::RefCell};

use ndarray::{Array1, Array2};
use rand::Rng;

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

    fn _calc_do_output_layer(&self, t: &Array2<f64>) -> Array2<f64> {
        let do_output: Array2<f64> = (-t) / &self.output_array;
        do_output
    }

}

pub trait LayerBase {
    fn weight(&self) -> &Array2<f64>;
    fn bias(&self) -> &Array1<f64>;
    fn output(&self) -> &Array2<f64>;
    fn input_grad(&self) -> &Array2<f64>;

    fn set_next_layer(&mut self, next_layer: Rc<RefCell<dyn LayerBase>>);
    fn set_input(&mut self, x: &Array2<f64>);
    fn set_answer(&mut self, t: &Array2<f64>);

    fn calc_output(&mut self, x: &Array2<f64>);
    // fn calc_delta(&mut self);

    /// weight, bias, inputの勾配を計算する
    fn calc_grad(&mut self);

    /// 保持している勾配情報を消費して重み、バイアスを更新する
    fn update(&mut self, lr: f64);
}


/// ソフトマックス関数をするやつ
pub struct OutputLayer {
    weight: Array2<f64>,
    bias: Array1<f64>,
    input: Array2<f64>,
    output: Array2<f64>,
    delta: Array2<f64>,
    weight_grad: Option<Array2<f64>>,
    bias_grad: Option<Array1<f64>>,
    input_grad: Option<Array2<f64>>,
    t: Option<Array2<f64>>,
}

impl OutputLayer {
    pub fn new(input_num: usize, output_num: usize, batch_num: usize) -> OutputLayer {
        OutputLayer {
            weight: Array2::zeros((input_num, output_num)),
            bias: Array1::from_elem(output_num, 1.0),
            input: Array2::zeros((batch_num, input_num)),
            output: Array2::zeros((batch_num, output_num)),
            delta: Array2::zeros((batch_num, output_num)),
            weight_grad: None,
            bias_grad: None,
            input_grad: None,
            t: None,
        }
    }

    pub fn new_rand(input_num: usize, output_num: usize, batch_num: usize, ) -> OutputLayer {
        let mut weight = Array2::zeros((input_num, output_num));
        weight.map_mut(|x: &mut f64| *x = Rng::gen(&mut rand::thread_rng()));
        let mut bias = Array1::zeros(output_num);
        bias.map_mut(|x: &mut f64| *x = Rng::gen(&mut rand::thread_rng()));

        println!("weight = {:?}", &weight);
        println!("bias = {:?}", &bias);

        OutputLayer {
            weight: weight,
            bias: bias,
            input: Array2::zeros((batch_num, input_num)),
            output: Array2::zeros((batch_num, output_num)),
            delta: Array2::zeros((batch_num, output_num)),
            weight_grad: None,
            bias_grad: None,
            input_grad: None,
            t: None,
        }
    }

    fn calc_delta(&mut self) {
        if let Some(t) = self.t.take() {
            self.delta = &self.output - t;
            // println!("delta = {:?}", &self.delta);
        } else {
            panic!("t is not set");
        }
    }

    fn set_testdata(&mut self, t: &Array2<f64>) {
        self.t.replace(t.clone());
    }
}

impl LayerBase for OutputLayer {
    fn input_grad(&self) -> &Array2<f64> {
        &self.input_grad.as_ref().unwrap()
    }

    fn bias(&self) -> &Array1<f64> {
        &self.bias
    }

    fn weight(&self) -> &Array2<f64> {
        &self.weight
    }

    fn output(&self) -> &Array2<f64> {
        &self.output
    }

    fn set_input(&mut self, x: &Array2<f64>) {
        self.input = x.clone();
    }

    fn set_next_layer(&mut self, _next_layer: Rc<RefCell<dyn LayerBase>>) {
        panic!("OutputLayer can not have next layer");
    }

    fn set_answer(&mut self, t: &Array2<f64>) {
        // println!("t = {:?}", &t);
        self.set_testdata(t);
    }

    fn calc_output(&mut self, x: &Array2<f64>) {
        println!("O weight = {:?}", &self.weight);
        println!("O bias = {:?}", &self.bias);
        println!("O input = {:?}", &x);
        self.set_input(x);
        let u = self.input.dot(&self.weight) + &self.bias;
        println!("O u = {:?}", &u);
        let output = act_fn::batch_softmax(&u);
        println!("O output = {:?}", &output);
        self.output = output;
    }

    /// 保持しているテストデータを消費して計算する
    fn calc_grad(&mut self) {
        self.calc_delta();

        let weight_grad = self.input.t().dot(&self.delta);
        self.weight_grad.replace(weight_grad);

        let bias_grad = self.delta.sum_axis(ndarray::Axis(0));
        self.bias_grad.replace(bias_grad);


        // println!("delta = {:?}", &self.delta);
        // println!("weight = {:?}", &self.weight);
        self.input_grad.replace(self.delta.dot(&self.weight.t()));
        // println!("input_grad = {:?}", &self.input_grad);
    }

    fn update(&mut self, lr: f64) {
        println!("ois");
        if let Some(weight_grad) = self.weight_grad.take() {
            if let Some(bias_grad) = self.bias_grad.take() {
                self.weight = &self.weight - lr * weight_grad;
                self.bias = &self.bias - lr * bias_grad;
            } else {
                panic!("bias_grad is not set");
            }
        } else {
            panic!("weight_grad is not set");
        }
    }
}


pub struct HiddenLayer {
    input: Array2<f64>,
    output: Array2<f64>,
    weight: Array2<f64>,
    bias: Array1<f64>,
    weight_grad: Option<Array2<f64>>,
    bias_grad: Option<Array1<f64>>,
    delta: Array2<f64>,
    input_grad: Option<Array2<f64>>,
    next_layer: Option<Rc<RefCell<dyn LayerBase>>>,
}

impl HiddenLayer {
    pub fn new(input_num: usize, output_num: usize, batch_num: usize) -> HiddenLayer {
        HiddenLayer {
            input: Array2::zeros((batch_num, input_num)),
            output: Array2::zeros((batch_num, output_num)),
            weight: Array2::zeros((input_num, output_num)),
            bias: Array1::from_elem(output_num, 1.0),
            weight_grad: None,
            bias_grad: None,
            delta: Array2::zeros((batch_num, output_num)),
            input_grad: None,
            next_layer: None,
        }
    }

    fn calc_delta(&mut self) {
        if let Some(next_layer) = self.next_layer.as_ref() {
            let output_grad: Array2<f64> = next_layer.borrow().input_grad().clone();
            let dy_du: Array2<f64> = &self.output * (Array2::from_elem(self.output.dim(), 1.0) - &self.output);

            self.delta = output_grad * dy_du;
        } else {
            panic!("next_layer is not set");
        }
    }

    pub fn set_next_layer(&mut self, next_layer: Rc<RefCell<dyn LayerBase>>) {
        assert!(next_layer.borrow().weight().shape()[0] == self.weight.shape()[1], "next_layer.borrow().weight().shape()[0] != self.weight.shape()[1]");
        self.next_layer.replace(next_layer);
    }
}

impl LayerBase for HiddenLayer {
    fn weight(&self) -> &Array2<f64> {
        &self.weight
    }

    fn bias(&self) -> &Array1<f64> {
        &self.bias
    }

    fn output(&self) -> &Array2<f64> {
        &self.output
    }

    fn set_input(&mut self, x: &Array2<f64>) {
        self.input = x.clone();
    }

    fn set_next_layer(&mut self, next_layer: Rc<RefCell<dyn LayerBase>>) {
        self.set_next_layer(next_layer);
    }

    fn set_answer(&mut self, _t: &Array2<f64>) {
        panic!("HiddenLayer can not have answer");
    }

    fn input_grad(&self) -> &Array2<f64> {
        &self.input_grad.as_ref().unwrap()
    }

    fn calc_output(&mut self, x: &Array2<f64>) {
        assert_eq!(x.shape()[1] , self.weight.shape()[0], "x.shape()[1] != self.weight.shape()[0]");
        
        println!("H weight = {:?}", &self.weight);
        println!("H bias = {:?}", &self.bias);
        println!("H input = {:?}", &x);
        self.set_input(x);
        let u = self.input.dot(&self.weight) + &self.bias;
        println!("H u = {:?}", &u);
        let output = u.map(|&x| act_fn::sigmoid(x));
        println!("H output = {:?}", &output);
        self.output = output;
    }

    fn calc_grad(&mut self) {
        self.calc_delta();


        println!("delta = {:?}", &self.delta);
        println!("input = {:?}", &self.input);
        let weight_grad = self.input.t().dot(&self.delta);
        println!("weight_grad = {:?}", &weight_grad);
        self.weight_grad.replace(weight_grad);

        let bias_grad = self.delta.sum_axis(ndarray::Axis(0));
        self.bias_grad.replace(bias_grad);

        self.input_grad.replace(self.delta.dot(&self.weight.t()));
    }

    fn update(&mut self, lr: f64) {
        println!("ois hid");
        if let Some(weight_grad) = self.weight_grad.take() {
            if let Some(bias_grad) = self.bias_grad.take() {
                println!("iketeru");
                println!("weight = {:?}", &self.weight);
                println!("weight_grad = {:?}", &weight_grad);
                self.weight = &self.weight - lr * weight_grad;
                println!("iketeru!");
                self.bias = &self.bias - lr * bias_grad;
                println!("iketeru!!");
            } else {
                panic!("bias_grad is not set");
            }
        } else {
            panic!("weight_grad is not set");
        }
    }
}