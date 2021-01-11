use crate::graph::Graph;
use crate::{uuid::Uuid, Float, FxHashMap, GraphRepr, NdArray};
use smallvec::alloc::borrow::Cow;
use smallvec::alloc::fmt::Formatter;
use std::cell::{RefCell, UnsafeCell};
use std::ops::Deref;

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
/// Unique ID assigned to a variable array in a `VariableEnvironment`
pub struct VariableID(pub(crate) usize);

impl From<usize> for VariableID {
    fn from(a: usize) -> VariableID {
        VariableID(a)
    }
}

impl From<VariableID> for usize {
    fn from(a: VariableID) -> usize {
        a.0
    }
}

impl std::fmt::Display for VariableID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub type Variable<F> = RefCell<NdArray<F>>;

pub struct VariableEnvironment<'name, F: Float> {
    pub(crate) variable_vec: Vec<Variable<F>>,
    pub(crate) variable_map: FxHashMap<FullName<'name>, VariableID>,
}

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct FullName<'name> {
    pub(crate) namespace_name: Cow<'name, str>,
    pub(crate) variable_name: Cow<'name, str>,
}

pub struct VariableSlot<'ns, 'env, 'name, F: Float> {
    namespace: &'ns mut VariableNamespaceMut<'env, 'name, F>,
}

pub struct NamedVariableSlot<'ns, 'env, 'name, F: Float, S: Into<String>> {
    namespace: &'ns mut VariableNamespaceMut<'env, 'name, F>,
    name: S,
}

pub struct DefaultVariableSlot<'env, 'name, F: Float> {
    env: &'env mut VariableEnvironment<'name, F>,
}

pub struct NamedDefaultVariableSlot<'env, 'name, F: Float, S: Into<String>> {
    env: &'env mut VariableEnvironment<'name, F>,
    name: S,
}

pub struct VariableNamespace<'env, 'name, F: Float> {
    pub(crate) env: &'env VariableEnvironment<'name, F>,
    pub(crate) namespace_name: &'static str,
}

pub struct VariableNamespaceMut<'env, 'name, F: Float> {
    pub(crate) env: &'env mut VariableEnvironment<'name, F>,
    pub(crate) namespace_name: &'static str,
}

impl<'name> FullName<'name> {
    pub(crate) fn new(namespace_name: &'static str, variable_name: Cow<'name, str>) -> Self {
        FullName {
            namespace_name: Cow::Borrowed(namespace_name),
            variable_name,
        }
    }
}

pub trait NamespaceTrait<F: Float> {
    fn name(&self) -> &'static str;

    fn env(&self) -> &VariableEnvironment<F>;

    #[inline]
    fn get_array_by_id(&self, vid: VariableID) -> &RefCell<NdArray<F>> {
        &self.env().variable_vec[vid.0]
    }

    #[inline]
    fn get_array_by_name<S: AsRef<str>>(&self, name: S) -> Option<&RefCell<NdArray<F>>> {
        let name = &FullName::new(self.name(), Cow::Borrowed(name.as_ref()));
        self.env()
            .variable_map
            .get(name)
            .map(|vid| &self.env().variable_vec[vid.0])
    }

    fn current_var_ids(&self) -> Vec<VariableID> {
        self.env()
            .variable_map
            .iter()
            .filter_map(|(v_name, &vid)| {
                if v_name.namespace_name == self.name() {
                    Some(vid)
                } else {
                    None
                }
            })
            .collect()
    }

    fn current_var_names(&self) -> Vec<&str> {
        self.env()
            .variable_map
            .iter()
            .filter_map(|(v_name, _)| {
                if v_name.namespace_name == self.name() {
                    Some(v_name.variable_name.deref())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl<'ns, 'env, 'name, F: Float, S: Into<String>> NamedVariableSlot<'ns, 'env, 'name, F, S> {
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(
            v,
            self.namespace.namespace_name,
            self.name.into(),
            self.namespace.env,
        )
    }
}

impl<'env, 'name, F: Float> DefaultVariableSlot<'env, 'name, F> {
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(v, "", Uuid::new_v4().to_string(), self.env)
    }

    pub fn with_name<S: Into<String>>(
        self,
        name: S,
    ) -> NamedDefaultVariableSlot<'env, 'name, F, S> {
        NamedDefaultVariableSlot {
            env: self.env,
            name,
        }
    }
}

impl<'env, 'name, F: Float, S: Into<String>> NamedDefaultVariableSlot<'env, 'name, F, S> {
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(v, "", self.name.into(), self.env)
    }
}

impl<'ns, 'env, 'name, F: Float> VariableSlot<'ns, 'env, 'name, F> {
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(
            v,
            self.namespace.namespace_name,
            Uuid::new_v4().to_string(),
            self.namespace.env,
        )
    }

    pub fn with_name<S: Into<String>>(self, name: S) -> NamedVariableSlot<'ns, 'env, 'name, F, S> {
        NamedVariableSlot {
            namespace: self.namespace,
            name,
        }
    }
}

fn register_variable<F: Float, D: ndarray::Dimension, S: Into<String>>(
    v: ndarray::Array<F, D>,
    namespace_name: &'static str,
    variable_name: S,
    env: &mut VariableEnvironment<F>,
) -> VariableID {
    let vid = FullName::new(namespace_name, Cow::Owned(variable_name.into()));
    let next_id = env.variable_vec.len().into();
    env.variable_map.insert(vid, next_id);
    env.variable_vec.push(RefCell::new(v.into_dyn()));
    next_id
}

impl<'env, 'name, F: Float> NamespaceTrait<F> for VariableNamespace<'env, 'name, F> {
    #[inline]
    fn name(&self) -> &'static str {
        self.namespace_name
    }
    #[inline]
    fn env(&self) -> &VariableEnvironment<F> {
        self.env
    }
}

impl<'env, 'name, F: Float> NamespaceTrait<F> for VariableNamespaceMut<'env, 'name, F> {
    #[inline]
    fn name(&self) -> &'static str {
        self.namespace_name
    }
    #[inline]
    fn env(&self) -> &VariableEnvironment<F> {
        self.env
    }
}

impl<'ns, 'env, 'name, F: Float> VariableNamespaceMut<'env, 'name, F> {
    pub fn slot(&'ns mut self) -> VariableSlot<'ns, 'env, 'name, F> {
        VariableSlot { namespace: self }
    }
}

impl<'env, 'name, F: Float> VariableEnvironment<'name, F> {
    pub fn new() -> VariableEnvironment<'name, F> {
        Self {
            variable_map: FxHashMap::default(),
            variable_vec: Vec::new(),
        }
    }

    pub fn slot(&'env mut self) -> DefaultVariableSlot<'env, 'name, F> {
        DefaultVariableSlot { env: self }
    }

    #[inline]
    pub fn namespace(
        &'env self,
        namespace_name: &'static str,
    ) -> VariableNamespace<'env, 'name, F> {
        VariableNamespace {
            namespace_name,
            env: self,
        }
    }

    #[inline]
    pub fn namespace_mut(
        &'env mut self,
        namespace_name: &'static str,
    ) -> VariableNamespaceMut<'env, 'name, F> {
        VariableNamespaceMut {
            namespace_name,
            env: self,
        }
    }

    #[inline]
    pub fn default_namespace(&'env self) -> VariableNamespace<'env, 'name, F> {
        self.namespace("")
    }

    #[inline]
    pub fn default_namespace_mut(&'env mut self) -> VariableNamespaceMut<'env, 'name, F> {
        self.namespace_mut("")
    }

    #[inline]
    pub fn get_variable(&self, vid: VariableID) -> &Variable<F> {
        &self.variable_vec[vid.0]
    }

    /// Creates a namespace for a computation graph.
    ///
    /// This is the only way to create [Graph](struct.Graph.html) instances.
    pub fn run<FN, R>(&'env mut self, f: FN) -> R
    where
        FN: FnOnce(&mut Graph<'env, 'name, F>) -> R,
    {
        let g = GraphRepr {
            node_set: UnsafeCell::new(Vec::with_capacity(256)),
            variable2node: RefCell::new(FxHashMap::default()),
        };
        let mut c = Graph {
            env_handle: self,
            inner: g,
        };
        f(&mut c)
    }
}

#[allow(unused)]
fn test() {
    let mut env = VariableEnvironment::<f32>::new();
    let _cur_names_ = env.default_namespace().current_var_names();

    env.run(|g| {
        let ns = g.env().default_namespace();
        let var = g.variable_map_by_name(&ns);

        let _v3_ = g.variable_by_name("a", &ns);
        let v = var["a"];
        let v2 = var["a"];
        let ones = g.zeros(&[1]) + v + v2;
        let _ = ones.eval(&[], &g);
    })
}
