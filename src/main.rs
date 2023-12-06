#[macro_use]
extern crate ndarray;

use ndarray::{arr2, Array, Array1, Array2, Zip};

fn main() {
    mat();
    test_perceptron();
}

fn mat() {
    // let a = arr2(&[[1, 2, 3],
    //     [4, 5, 6]]);

    // println!("a = \n{:?}", &a);

    // assert_eq!(a.get([0,1]), Some(&2));
    // assert_eq!(a.get((0,3)), None);
    // assert_eq!(a[(1, 2)], 6);
    // assert_eq!(a[(1, 1)], 5);

    // let mut index: i16 = 0;
    // for e in a.iter() {
    //     index += 1;
    //     println!("e{} = {}", index, e);
    // }

    // let mut m: Array2<f64> = Array::zeros((10, 10));
    // println!("m = \n{:?}", &m);
    // let mut fill_val = 1.0;
    // for mut row in m.rows_mut() {
    //     row.fill(fill_val);
    //     fill_val += 1.5;
    // }
    // println!("m = \n{:?}", &m);


    // type M = Array2<f64>;
    // let mut a: M = M::zeros((12, 8));
    // let b = M::from_elem(a.dim(), 1.0);
    // let c = M::from_elem(a.dim(), 2.0);
    // let d = M::from_elem(a.dim(), 3.0);

    // Zip::from(&mut a)
    //     .and(&b)
    //     .and(&c)
    //     .and(&d)
    //     .for_each(|w, &x, &y, &z| {
    //         *w = x + y * z ;
    //     });
    // println!("{:?}", &a);

    // let mut t = Array::zeros((5, 5, 5));
    // let mut idx = 0;
    // for elm in t.iter_mut() {
    //     *elm = idx;
    //     idx += 1;
    // }
    // println!("{:?}", t.slice(s![.., 0..2, 2..4]));
}

fn test_perceptron() {
    let x: Array1<f64> = array![1.0, 1.0];
    println!("x = {:?}", &x);
    println!("and = {}", and(&x));
    println!("or = {}", or(&x));
    println!("nand = {}", nand(&x));
    println!("xor = {}", xor(&x));
}

fn over(x: f64, theta: f64) -> f64 {
    return if x >= theta {1.0} else {0.0};
}

pub fn and(x: &Array1<f64>) -> f64 {
    let w: Array1<f64> = array![0.5, 0.5];
    let theta: f64 = 0.7;
    over(w.dot(x), theta)
}

pub fn or(x: &Array1<f64>) -> f64 {
    let w: Array1<f64> = array![0.5, 0.5];
    let theta: f64 = 0.3;
    over(w.dot(x), theta)
}

pub fn nand(x: &Array1<f64>) -> f64 {
    let w: Array1<f64> = array![-0.5, -0.5];
    let theta: f64 = -0.7;
    over(w.dot(x), theta)
}

pub fn xor(x: &Array1<f64>) -> f64 {
    let y1 = or(x);
    let y2 = nand(x);
    and(&array![y1, y2])
}