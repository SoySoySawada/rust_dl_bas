use ndarray::Array2;

use std::{rc::Rc, cell::RefCell};

use crate::layer::Layer;

pub struct Network {
    pub layers: Vec<Rc<RefCell<Layer>>>,
}

impl Network {
    pub fn new() -> Network {
        Network {
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer: Rc<RefCell<Layer>>) {
        let prev_layer = self.layers.last();

        if let Some(prev_layer) = prev_layer {
            layer.borrow_mut().set_prev_layer(Rc::clone(prev_layer));
            prev_layer.borrow_mut().set_next_layer(Rc::clone(&layer));
        }

        self.layers.push(layer);
    }

    pub fn train(&mut self, x: &Array2<f64>, t: &Array2<f64>) {
        self.calc_result(x);

        let last_layer = self.layers.last().unwrap();
        last_layer.borrow_mut().calc_du_output_layer(t);
        println!("last layer du = {:?}", &last_layer.as_ref().borrow_mut().du);

        for layer in self.layers.iter().rev().skip(1) {
            layer.borrow_mut().calc_du_hidden_layer();
            println!("layer du = {:?}", &layer.as_ref().borrow_mut().du);
        }

        println!("layer0 du = {:?}", &self.layers[0].as_ref().borrow_mut().du);
        println!("layer1 du = {:?}", &self.layers[1].as_ref().borrow_mut().du);
        // println!("layer2 du = {:?}", &self.layers[2].as_ref().borrow_mut().du);

        let learn_rate = 0.05;
        let first_layer = self.layers.first().unwrap();
        first_layer.borrow_mut().update_first_layer(x, learn_rate);
        for layer in self.layers.iter().skip(1) {
            layer.borrow_mut().update_other_layer(learn_rate);
        }

        println!("train now");
    }

    pub fn calc_result(&mut self, x: &Array2<f64>) {
        let first_layer = self.layers.first().unwrap();

        first_layer.borrow_mut().calc_output_first_layer(x);

        for layer in self.layers.iter().skip(1) {
            layer.borrow_mut().calc_output_other_layer();
        }

        println!("result = {:?}", self.result());
    }

    pub fn result(&self) -> Array2<f64> {
        let last_layer = self.layers.last().unwrap();
        last_layer.borrow().output_array.clone()
    }

}