use ndarray::{Array1, Array2, Axis};


#[derive(Debug, Clone)]
pub struct Neural {
    hidden_weight: Array2<f64>,
    output_weight: Array2<f64>,
    hidden_momentum : Array2<f64>,
    output_momentum : Array2<f64>,
    error: Option<Array1<f64>>,
}

impl Neural {
    pub fn new(input_num: usize, hidden_num: usize, output_num: usize) -> Self {
        Neural {
            hidden_weight: Array2::zeros((hidden_num, input_num + 1)),
            output_weight: Array2::zeros((output_num, hidden_num + 1)),
            hidden_momentum: Array2::zeros((hidden_num, input_num + 1)),
            output_momentum: Array2::zeros((output_num, hidden_num + 1)),
            error: None,
        }
    }

    pub fn init(&mut self) {
        self.hidden_weight = self.hidden_weight.map(|_| rand::random::<f64>());
        self.output_weight = self.output_weight.map(|_| rand::random::<f64>());
    }

    pub fn error(&self) -> &Array1<f64> {
        self.error.as_ref().unwrap()
    }

    /// epoch: エポック数
    pub fn train(&mut self, input: Array2<f64>, t: Array2<f64>, epsilon: f64, mu: f64, epoch: usize) {
        self.error.replace(Array1::zeros(epoch));
        let input_num = input.shape()[0];

        
        for e in 0..epoch {
            for i in 0..input_num {
                let x: Array2<f64> = input.slice(s![i, ..]).insert_axis(Axis(0)).to_owned();
                let t: Array2<f64> = t.slice(s![i, ..]).insert_axis(Axis(0)).to_owned();
                
                self.update_weight(x, t, epsilon, mu);
            }
            self.error.as_mut().unwrap()[e] = self.calc_error(&input, &t);
        }
    }


    pub fn predict(&self, input: Array2<f64>) -> Array2<f64> {
        let n = input.shape()[0];
        // let cost: Array2<i32> = Array2::zeros((1, n));
        let mut y: Array2<f64> = Array2::zeros((n, input.shape()[1]));

        for i in 0..n {
            let x = input.slice(s![i, ..]).insert_axis(Axis(0)).to_owned();
            let (_, output_layer_output) = self.forward(&x);

            y.slice_mut(s![i, ..]).assign(&output_layer_output.remove_axis(Axis(0)));
        }
        y
    }

    /// input: 1 * n 行列
    /// t: 1 * m 行列
    fn update_weight(&mut self, input: Array2<f64>, t: Array2<f64>, epsilon: f64, mu: f64) {
        let (hidden_layer_output, output_layer_output) = self.forward(&input);

        // update output layer weight
        let output_weight_begore = self.output_weight.clone();
        let output_delta = (&output_layer_output - t) * &output_layer_output * (1.0 - &output_layer_output);
        let output_delta_len = output_delta.len();
        let output_delta_col = output_delta.clone().into_shape((output_delta_len, 1)).unwrap();
        let output_weight_grad: Array2<f64> = output_delta_col.dot(&concatenate![Axis(1), Array2::ones((1, 1)), hidden_layer_output.clone()]);
        self.output_weight = &self.output_weight - ( epsilon * output_weight_grad - mu * &self.output_momentum);
        self.output_momentum = &self.output_weight - output_weight_begore;

        // update hidden layer weight
        let hidden_weight_before = self.hidden_weight.clone();
        let hidden_delta = (&output_delta.dot(&self.output_weight.slice(s![.., 1..]))) * &hidden_layer_output * (1.0 - &hidden_layer_output);
        let hidden_delta_len = hidden_delta.len();
        let hidden_delta_col = hidden_delta.into_shape((hidden_delta_len, 1)).unwrap();
        let hidden_weight_grad: Array2<f64> = hidden_delta_col.dot(&concatenate![Axis(1), Array2::ones((1, 1)), input.clone()]);
        self.hidden_weight = &self.hidden_weight - (epsilon * hidden_weight_grad);
        self.hidden_momentum = &self.hidden_weight - hidden_weight_before;
    }

    /// x: 入力(行 行列)
    /// return: (hidden layer output, output layer output)
    fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // numpy.r_ → stack![Axis(1), ...]
        let stack_array: Array2<f64> = Array2::ones((1, 1));
        let a = concatenate![Axis(1), stack_array, x.clone()];
        // let hidden_layer_output = Self::sigmoid(self.hidden_weight.dot(&a)); // pythonの記述と異なる
        let hidden_layer_output = Self::sigmoid(a.dot(&self.hidden_weight.t()));

        let b = concatenate![Axis(1), stack_array, hidden_layer_output];
        // let output_layer_output = Self::sigmoid(self.output_weight.dot(&b)); // pythonの記述と異なる
        let output_layer_output = Self::sigmoid(b.dot(&self.output_weight.t()));

        (hidden_layer_output, output_layer_output)
    }

    fn sigmoid(arr: Array2<f64>) -> Array2<f64> {
        arr.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// 二乗誤差の計算
    fn calc_error(&mut self, input: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let n = input.shape()[0];
        let mut err = 0.;
        for i in 0..n {
            let x = input.slice(s![i, ..]).insert_axis(Axis(0)).to_owned();
            let t = t.slice(s![i, ..]).insert_axis(Axis(0)).to_owned();

            let (_, output_layer_output) = self.forward(&x);
            err += (output_layer_output - t).map(|x| x.powi(2)).sum();
        }
        err
    }

    // これまでの重みの実装方法(行：入力数, 列：出力数)とは異なるので注意
}

#[cfg(test)]
mod test_neural {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sigmoid() {
        let sig_arr = Neural::sigmoid(arr2(&[[0.0, 0.5], [1.0, 2.0]]));
        let a11 = 1.0 / (1.0 + (-0.0 as f64).exp());
        let a12 = 1.0 / (1.0 + (-0.5 as f64).exp());
        let a21 = 1.0 / (1.0 + (-1.0 as f64).exp());
        let a22 = 1.0 / (1.0 + (-2.0 as f64).exp());
        assert_eq!(sig_arr, arr2(&[[a11, a12], [a21, a22]]));
    }

    #[test]
    fn test_forward() {
        let input = arr2(&[[0.0, 1.0]]);

        let input_size = input.shape()[1];
        let hidden_size = 2;
        let output_size = 2;
        
        let hidden_weight = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let output_weight = arr2(&[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]);

        let mut neural = Neural::new(input_size, hidden_size, output_size);

        assert_eq!(neural.hidden_weight, arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]));
        assert_eq!(neural.output_weight, arr2(&[[0.0, 0.0, 0.], [0.0, 0.0, 0.]]));

        neural.hidden_weight = hidden_weight.clone();
        neural.output_weight = output_weight.clone();

        assert_eq!(neural.hidden_weight, hidden_weight);
        assert_eq!(neural.output_weight, output_weight);

        let h_u1 = 0.1 * 1.0 + 0.2 * 0.0 + 0.3 * 1.0;
        let h_u2 = 0.4 * 1.0 + 0.5 * 0.0 + 0.6 * 1.0;
        let h_y1 = 1.0 / (1.0 + (-h_u1 as f64).exp());
        let h_y2 = 1.0 / (1.0 + (-h_u2 as f64).exp());
        let o_u1 = 0.7 * 1.0 + 0.8 * h_y1 + 0.9 * h_y2;
        let o_u2 = 1.0 * 1.0 + 1.1 * h_y1 + 1.2 * h_y2;
        let o_y1 = 1.0 / (1.0 + (-o_u1 as f64).exp());
        let o_y2 = 1.0 / (1.0 + (-o_u2 as f64).exp());

        let (hidden_layer_output, output_layer_output) = neural.forward(&input);
        
        assert_eq!(hidden_layer_output, arr2(&[[h_y1, h_y2]]));
        assert_eq!(output_layer_output, arr2(&[[o_y1, o_y2]]));

        println!("hidden_layer_output = {:?}", &hidden_layer_output);
        println!("output_layer_output = {:?}", &output_layer_output);
    }

    #[test]
    fn test_update_weight() {
        let input = arr2(&[[0.0, 1.0]]);
        let t = arr2(&[[0.0, 1.0]]);

        let input_size = input.shape()[1];
        let hidden_size = 2;
        let output_size = 2;
        
        let hidden_weight = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let output_weight = arr2(&[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]);

        let mut neural = Neural::new(input_size, hidden_size, output_size);

        neural.hidden_weight = hidden_weight.clone();
        neural.output_weight = output_weight.clone();

        let epsilon = 0.1;
        let mu = 0.1;

        let h_u1 = 0.1 * 1.0 + 0.2 * 0.0 + 0.3 * 1.0;
        let h_u2 = 0.4 * 1.0 + 0.5 * 0.0 + 0.6 * 1.0;
        let h_y1 = 1.0 / (1.0 + (-h_u1 as f64).exp());
        let h_y2 = 1.0 / (1.0 + (-h_u2 as f64).exp());
        let o_u1 = 0.7 * 1.0 + 0.8 * h_y1 + 0.9 * h_y2;
        let o_u2 = 1.0 * 1.0 + 1.1 * h_y1 + 1.2 * h_y2;
        let o_y1 = 1.0 / (1.0 + (-o_u1 as f64).exp());
        let o_y2 = 1.0 / (1.0 + (-o_u2 as f64).exp());

        let o_de_dy = arr2(&[[o_y1 - &t[(0, 0)], o_y2 - &t[(0, 1)]]]);
        let o_de_du = arr2(&[[o_de_dy[(0, 0)] * o_y1 * (1.0 - o_y1), o_de_dy[(0, 1)] * o_y2 * (1.0 - o_y2)]]);
        let o_w_delta0_1 = o_de_du[(0, 0)] * 1.0;
        let o_w_delta0_2 = o_de_du[(0, 1)] * 1.0;
        let o_w_delta1_1 = o_de_du[(0, 0)] * h_y1;
        let o_w_delta1_2 = o_de_du[(0, 1)] * h_y1;
        let o_w_delta2_1 = o_de_du[(0, 0)] * h_y2;
        let o_w_delta2_2 = o_de_du[(0, 1)] * h_y2;

        let h_de_dy: Array2<f64> = arr2(&[[ o_de_du[(0, 0)] * &output_weight[(0, 1)] + o_de_du[(0, 1)] * &output_weight[(1, 1)], 
                                                o_de_du[(0, 0)] * &output_weight[(0, 2)] + o_de_du[(0, 1)] * &output_weight[(1, 2)]]]);
        let h_de_du: Array2<f64> = arr2(&[[h_de_dy[(0, 0)] * h_y1 * (1.0 - h_y1), h_de_dy[(0, 1)] * h_y2 * (1.0 - h_y2)]]);
        let _h_w_delta0_1 = h_de_du[(0, 0)] * 1.0;
        let _h_w_delta0_2 = h_de_du[(0, 1)] * 1.0;
        let h_w_delta1_1 = h_de_du[(0, 0)] * 0.0;
        let h_w_delta1_2 = h_de_du[(0, 1)] * 0.0;
        let _h_w_delta2_1 = h_de_du[(0, 0)] * 1.0;
        let _h_w_delta2_2 = h_de_du[(0, 1)] * 1.0; 

        neural.update_weight(input, t, epsilon, mu);
        assert_eq!(neural.output_weight[(0, 0)], output_weight[(0, 0)] - o_w_delta0_1 * epsilon);
        assert_eq!(neural.output_weight[(0, 1)], output_weight[(0, 1)] - o_w_delta1_1 * epsilon);
        assert_eq!(neural.output_weight[(0, 2)], output_weight[(0, 2)] - o_w_delta2_1 * epsilon);
        assert_eq!(neural.output_weight[(1, 0)], output_weight[(1, 0)] - o_w_delta0_2 * epsilon);
        assert_eq!(neural.output_weight[(1, 1)], output_weight[(1, 1)] - o_w_delta1_2 * epsilon);
        assert_eq!(neural.output_weight[(1, 2)], output_weight[(1, 2)] - o_w_delta2_2 * epsilon);
        // assert_eq!(neural.output_momentum, arr2(&[[-o_w_delta0_1 * epsilon, -o_w_delta1_1 * epsilon, -o_w_delta2_1 * epsilon], 
        //                                           [-o_w_delta0_2 * epsilon, -o_w_delta1_2 * epsilon, -o_w_delta2_2 * epsilon]]));
        // assert_eq!(neural.output_momentum, neural.output_weight - output_weight);


        // assert_eq!(neural.hidden_weight[(0, 0)], hidden_weight[(0, 0)] - h_w_delta0_1 * epsilon);
        assert_eq!(neural.hidden_weight[(0, 1)], hidden_weight[(0, 1)] - h_w_delta1_1 * epsilon);
        // assert_eq!(neural.hidden_weight[(0, 2)], hidden_weight[(0, 2)] - h_w_delta2_1 * epsilon);
        // assert_eq!(neural.hidden_weight[(1, 0)], hidden_weight[(1, 0)] - h_w_delta0_2 * epsilon);
        assert_eq!(neural.hidden_weight[(1, 1)], hidden_weight[(1, 1)] - h_w_delta1_2 * epsilon);
        // assert_eq!(neural.hidden_weight[(1, 2)], hidden_weight[(1, 2)] - h_w_delta2_2 * epsilon);
        assert_eq!(neural.hidden_momentum, neural.hidden_weight - hidden_weight);
    }
}