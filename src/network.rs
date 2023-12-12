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
}