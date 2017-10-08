extern crate ndarray;
extern crate fnv;

use self::fnv::FnvHashMap;
use ndarray_ext::NdArray;
use ops;
use std::cmp::Ordering;
use std::collections::hash_set::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;


/// Symbolic multi-dimensional array which supports
/// efficient gradient computation.
pub struct Tensor(pub Rc<RawTensor>);

pub struct RawTensor {
    /// Operation created this node
    pub op: Box<ops::Op>,

    /// References to immediate predecessors.
    pub inputs: Vec<Tensor>,

    /// Rank number for topological ordering
    pub top_rank: usize,
}


impl Tensor {
    #[doc(hidden)]
    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool
    {
        self.inputs.is_empty()
    }

    #[doc(hidden)]
    #[inline]
    pub fn visit_once<F>(&self, f: &mut F)
    where
        F: FnMut(&Tensor) -> (),
    {
        self.run_visit_once(f, &mut HashSet::new())
    }

    #[inline]
    fn run_visit_once<F>(&self, f: &mut F, visited: &mut HashSet<Tensor>)
    where
        F: FnMut(&Tensor) -> (),
    {
        if visited.contains(self) {
            return; // exit early
        } else {
            visited.insert(self.clone()); // first visit
        }

        f(self);

        for child in &(*self).inputs {
            child.run_visit_once(f, visited)
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn visit<F>(&self, f: &mut F)
    where
        F: FnMut(&Tensor) -> (),
    {
        f(self);

        for child in &(*self).inputs {
            child.visit(f)
        }
    }
}

impl Ord for Tensor {
    /// Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering
    {
        self.top_rank.cmp(&other.top_rank)
    }
}

impl PartialOrd for Tensor {
    /// Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
    {
        Some(self.cmp(other))
    }
}

// empty implementation
impl Eq for Tensor {}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool
    {
        // compare addresses on the heap
        Rc::ptr_eq(&self.0, &other.0)
    }
}

// empty implementation
impl Hash for Tensor {
    fn hash<H: Hasher>(&self, _: &mut H)
    {
    }
}

// data is not cloned; only reference count is incremented.
impl Clone for Tensor {
    fn clone(&self) -> Tensor
    {
        Tensor(self.0.clone())
    }
}

impl Deref for Tensor {
    type Target = Rc<RawTensor>;
    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl DerefMut for Tensor {
    fn deref_mut<'a>(&'a mut self) -> &'a mut Rc<RawTensor>
    {
        &mut self.0
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let input_names = self.0
            .inputs
            .iter()
            .map(|a| a.op.name().to_string())
            .collect::<Vec<String>>();
        write!(
            f,
            "op: {}\ninputs: {:?}\n",
            self.0.op.name(),
            input_names.as_slice()
        )
    }
}


#[doc(hidden)]
#[inline]
pub fn eval_tensors(
    tensors: &[Tensor],
    variables: &mut FnvHashMap<Tensor, NdArray>,
    memo: &mut FnvHashMap<Tensor, NdArray>,
) -> Vec<NdArray>
{
    // run graph
    for t in tensors.iter() {
        ::topology::perform_eval(t, variables, memo, true, 0);
    }

    // extracts target arrays
    let mut evaluated_arrays = Vec::with_capacity(tensors.len());

    for (i, t) in tensors.iter().enumerate() {
        // Need to handle cases where multiple gradient nodes
        // share an output array, and `t` is a variable.
        // (Safe unwrapping is guaranteed by ::topology::symbolic_gradients)
        let contains = tensors[i + 1..].contains(t);
        let in_memo = memo.contains_key(t);
        match (contains, in_memo) {
            (true, true) => evaluated_arrays.push(memo.get(t).unwrap().clone()),
            (true, false) => evaluated_arrays.push(variables.get(t).unwrap().clone()),
            (false, true) => evaluated_arrays.push(memo.remove(t).unwrap()),
            (false, false) => evaluated_arrays.push(variables.get(t).unwrap().clone()),
        }
    }

    evaluated_arrays
}
