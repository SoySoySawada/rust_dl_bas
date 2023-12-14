use ndarray::Array2;

use std::{rc::Rc, cell::RefCell};

use crate::layer::LayerBase;

pub struct Network {
    pub layers: Vec<Rc<RefCell<dyn LayerBase>>>,
}

impl Network {
    pub fn new() -> Network {
        Network {
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer: Rc<RefCell<dyn LayerBase>>) {
        if let Some(prev_layer) = self.layers.last_mut() {
            prev_layer.borrow_mut().set_next_layer(Rc::clone(&layer));
        }

        self.layers.push(layer);
    }

    pub fn train(&mut self, x: &Array2<f64>, t: &Array2<f64>) {
        self.calc_result(x);


        self.layers.last().unwrap().borrow_mut().set_answer(t);

        println!("set answer");

        for layer in self.layers.iter().rev() {
            layer.borrow_mut().calc_grad();
        }
        
        println!("calc grad");

        let learn_rate = 0.001;
        for layer in self.layers.iter() {
            layer.borrow_mut().update(learn_rate);
        }
        println!("update");
    }

    pub fn calc_result(&mut self, x: &Array2<f64>) {
        let mut input = x.clone();
        for layer in self.layers.iter() {
            let mut layer_ref = layer.borrow_mut();
            println!("input = {:?}", &input);
            layer_ref.calc_output(&input);
            input = layer_ref.output().clone();
        }
        println!("calc result end");
    }

    pub fn result(&self) -> Array2<f64> {
        let last_layer = self.layers.last().unwrap();
        last_layer.borrow().output().clone()
    }

}
