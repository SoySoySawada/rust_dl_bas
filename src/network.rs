use std::rc::Rc;

use crate::layer;

pub struct Network {
    pub layers: Vec<Rc<layer::Layer>>,
}

impl Network {
    pub fn new() -> Network {
        Network {
            layers: Vec::new(),
        }
    }

    // fn add_layer(&mut self, layer: Rc<layer::Layer>) {
    //     let prev_layer = Rc::clone(self.layers.last_mut().unwrap());
    //     let add_layer = Rc::clone(&layer);

    //     if self.layers.len() > 0 {
    //         add_layer.set_prev_layer(prev_layer);
    //         prev_layer.set_next_layer(add_layer);
    //     }

    //     self.layers.push(add_layer);
    // }
}