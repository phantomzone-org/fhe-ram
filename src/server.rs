use crate::{client::EvaluationKeys, parameters::Parameters, ram::Ram};

pub struct Server {
    params: Parameters,
    ram: Ram,
    evk: EvaluationKeys,
}
