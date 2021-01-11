//! Defining things related to `ag::Graph`.

use crate::variable::VariableID;
use crate::variable::{FullName, NamespaceTrait};
use crate::{tensor::Tensor, tensor::TensorInternal, Float, FxHashMap, VariableEnvironment};
use smallvec::alloc::borrow::Cow;
use std::fmt;
use std::ops::Deref;
use std::{cell::RefCell, cell::UnsafeCell, collections::HashMap};

type TensorID = usize;

pub struct GraphRepr<F: Float> {
    pub(crate) node_set: UnsafeCell<Vec<TensorInternal<F>>>,
    pub(crate) variable2node: RefCell<FxHashMap<VariableID, TensorID>>,
}

impl<'t, 'g, 'e, F: Float> GraphRepr<F> {
    #[inline]
    pub(crate) fn install(&'g self, mut node: TensorInternal<F>) -> TensorID {
        unsafe {
            let inner = &mut *self.node_set.get();
            let id = inner.len();
            node.id = id;
            inner.push(node);
            id
        }
    }

    // `i` must be an id returned by Graph::install
    #[inline(always)]
    pub(crate) unsafe fn access_inner(&self, i: TensorID) -> &'t TensorInternal<F> {
        &(*self.node_set.get())[i]
    }

    // `i` must be an id returned by Graph::install
    #[inline(always)]
    pub(crate) unsafe fn access_inner_mut(&self, i: TensorID) -> &'t mut TensorInternal<F> {
        &mut (*self.node_set.get())[i]
    }

    #[inline(always)]
    pub(crate) fn tensor(&'g self, id: TensorID) -> Tensor<'g, F> {
        Tensor { id, graph: self }
    }

    #[inline]
    pub fn variable_by_id(&'g self, vid: VariableID) -> Tensor<'g, F> {
        let tid = {
            let temp = self.variable2node.borrow();
            temp.get(&vid).cloned()
        };
        if let Some(tid) = tid {
            // use existing tensor
            self.tensor(tid)
        } else {
            // allocate new tensor
            let allocated = Tensor::builder(self)
                .set_variable(vid)
                .build(self, crate::ops::basic_source_ops::Variable);
            // register vid -> tid map
            self.variable2node.borrow_mut().insert(vid, allocated.id);
            allocated
        }
    }

    #[inline]
    pub fn variable_by_name<S: AsRef<str>>(
        &self,
        name: S,
        namespace: &impl NamespaceTrait<F>,
    ) -> Tensor<F> {
        let full_name = &FullName::new(namespace.name(), Cow::Borrowed(name.as_ref()));
        if let Some(&vid) = namespace.env().variable_map.get(full_name) {
            // find VariableID
            self.variable_by_id(vid)
        } else {
            let ns = namespace.name();
            if ns == "" {
                panic!(
                    "variable array not found in default namespace: {}",
                    name.as_ref()
                )
            } else {
                panic!("variable array not found in `{}`: {}", ns, name.as_ref())
            }
        }
    }

    pub fn variable_map_by_id(
        &'g self,
        ctx: &'g VariableEnvironment<F>,
    ) -> HashMap<VariableID, Tensor<'g, F>> {
        (0..ctx.variable_vec.len())
            .map(|vid| (vid.into(), self.variable_by_id(vid.into())))
            .collect()
    }

    pub fn variable_map_by_name(
        &'g self,
        ns: &'g impl NamespaceTrait<F>,
    ) -> HashMap<&str, Tensor<'g, F>> {
        // reduce to target namespace
        let var_id_list = ns
            .env()
            .variable_map
            .iter()
            .filter_map(|ent| {
                // filter out other namespaces
                if &ent.0.namespace_name == ns.name() {
                    Some((*ent.1, ent.0.variable_name.as_ref()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // resolve
        var_id_list
            .into_iter()
            .map(|(vid, name)| (name, self.variable_by_id(vid)))
            .collect()
    }
}

impl<T: Float> fmt::Debug for GraphRepr<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            let set = &*self.node_set.get();
            let mut buf = format!("graph size: {}\n", set.len());
            for node in set {
                buf += format!("{}\n", node).as_str();
            }
            write!(f, "{}", buf)
        }
    }
}

/// Creates and runs a computation graph.
///
/// See also [Graph](struct.Graph.html).
/// ```
/// use autograd as ag;
/// use ag::ndarray;
///
/// let grad = ag::run(|graph| {
///     let x = graph.placeholder(&[]);
///     let y = graph.placeholder(&[]);
///     let z = 2.*x*x + 3.*y + 1.;
///
///     // dz/dx (symbolic):
///     let grad = &graph.grad(&[z], &[x])[0];
///
///     // Evaluate dz/dx when x=3:
///     grad.eval(&[x.given(ndarray::arr0(3.0).view())], graph).unwrap()
/// });
/// assert_eq!(grad, ndarray::arr0(12.0).into_dyn());
/// ```
pub fn run<F, FN, R>(f: FN) -> R
where
    F: Float,
    FN: FnOnce(&mut Graph<F>) -> R,
{
    let env_handle = &mut VariableEnvironment::new();
    let graph_internal = GraphRepr {
        node_set: UnsafeCell::new(Vec::with_capacity(512)),
        variable2node: RefCell::new(FxHashMap::default()),
    };
    let mut g = Graph {
        env_handle,
        inner: graph_internal,
    };
    f(&mut g)
}

/// Creates a scope for a computation graph.
///
/// Prefer to use [`run`] instead, as that is more flexible.
/// This function is kept for backwards compatibility.
pub fn with<F, FN>(f: FN)
where
    F: Float,
    FN: FnOnce(&mut Graph<F>),
{
    run(f);
}

/// Generator of `Tensor` objects.
///
/// Use [run] or [VariableEnvironment::run] to instantiate this.
pub struct Graph<'env, 'name, F: Float> {
    pub(crate) env_handle: &'env mut VariableEnvironment<'name, F>,
    pub(crate) inner: GraphRepr<F>,
}

impl<'env, 'name, F: Float> Graph<'env, 'name, F> {
    /// Returns the current VariableEnvironment
    #[inline]
    pub fn env(&self) -> &VariableEnvironment<F> {
        self.env_handle
    }

    /// Get or create a Tensor associated with the given `VariableID`.
    #[inline]
    pub fn variable_by_id(&self, vid: VariableID) -> Tensor<F> {
        self.inner.variable_by_id(vid)
    }

    #[inline]
    pub(crate) fn internal(&self) -> &GraphRepr<F> {
        &self.inner
    }
}

impl<'env, 'name, F: Float> Deref for Graph<'env, 'name, F> {
    type Target = GraphRepr<F>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub trait AsGraphRepr<F: Float> {
    fn as_graph_repr(&self) -> &GraphRepr<F>;
}

impl<F: Float> AsGraphRepr<F> for GraphRepr<F> {
    #[inline]
    fn as_graph_repr(&self) -> &GraphRepr<F> {
        self
    }
}

impl<F: Float> AsGraphRepr<F> for Graph<'_, '_, F> {
    #[inline]
    fn as_graph_repr(&self) -> &GraphRepr<F> {
        &self.inner
    }
}

#[inline]
pub(crate) fn assert_same_graph<F: Float>(a: &impl AsGraphRepr<F>, b: &impl AsGraphRepr<F>) {
    assert_eq!(
        a.as_graph_repr() as *const _,
        b.as_graph_repr() as *const _,
        "Detected tensors belonging to different graphs"
    );
}

#[test]
#[should_panic]
fn test_mixed_graph() {
    crate::VariableEnvironment::<f32>::new().run(|g| {
        let a = g.zeros(&[1]);
        crate::VariableEnvironment::<f32>::new().run(|g2| {
            let b = g2.zeros(&[1]);
            let _ = a + b;
        });
    });
}
