import inspect
import weakref
from numpy import ndarray, array_equal


"""
========================================
Metaclasses / classes for autoreloading
========================================

Suppose a class `C` is defined in module `c_module`, and `c` is an instance
of `C`. During an interactive session, suppose the user reloads `c_module`. 
Usually, the object `c` remains an instance of the outdated class. This
patterns allows `c` to be "rebased" upon the reload of `c_module`, so that
it is now an instance of the new version of `C`.



This is taken from:
    <http://bytes.com/topic/python/answers/...
        608972-automatic-reloading-metaclasses-pickle>
Which in turn is adapted from:
    <http://code.activestate.com/recipes/160164/>


To use, simply inherit from AutoReloader.

"""


class MetaInstanceTracker(type):

    def __init__(cls, name, bases, ns):
        super(MetaInstanceTracker, cls).__init__(name, bases, ns)
        cls.__instance_refs__ = []

    def __instances__(cls):
        instances = []
        validrefs = []
        for ref in cls.__instance_refs__:
            instance = ref()
            if instance is not None:
                instances.append(instance)
                validrefs.append(ref)
        cls.__instance_refs__ = validrefs
        return instances


class MetaAutoReloader(MetaInstanceTracker):

    def __init__(cls, name, bases, ns):
        super(MetaAutoReloader, cls).__init__(name, bases, ns)
        f = inspect.currentframe().f_back
        for d in [f.f_locals, f.f_globals]:
            if name in d:
                old_class = d[name]
                for instance in old_class.__instances__():
                    instance.__change_class__(cls)
                    cls.__instance_refs__.append(weakref.ref(instance))
                for subcls in old_class.__subclasses__():
                    newbases = []
                    for base in subcls.__bases__:
                        if base is old_class:
                            newbases.append(cls)
                        else:
                            newbases.append(base)
                    subcls.__bases__ = tuple(newbases)
                break


class InstanceTracker(object):

    __metaclass__ = MetaInstanceTracker

    def __new__(cls, *args, **kwargs):
        self = super( InstanceTracker, cls).__new__(cls, *args, **kwargs)
        cls.__instance_refs__.append(weakref.ref(self))
        return self

    def __reduce_ex__(self, proto):
        return super(InstanceTracker, self).__reduce_ex__(2)


class AutoReloader(InstanceTracker):

    __metaclass__ = MetaAutoReloader

    def __change_class__(self, new_class):
        self.__class__ = new_class


"""
========================================
Metaclasses / classes for autocaching
========================================

These classes allow certain attributes to be cached when they are first 
computed. The caching remains until one of their "ancestors" is overwritten
via a call to `__setattr__`. 

"""


class cached( object ):
    """Decorator creating a cached attribute.

    This decorator should be used on a function definition which appears in the definition of a
    class inheriting from 'AutoCacher'. Example:

    >>> class Objective( AutoCacher ):
    >>>    @cached
    >>>    def f( x, y ):
    >>>        return 2 * x + y

    >>> obj = Objective( x = 3, y = 4 )
    >>> print obj.f

    Whenever the x or y attributes change, the cached value of f will be discarded and f will be
    recomputed from the function definition as it is accessed.

    Note: All parameters of the function definition are taken as the names of attributes of the
    class instance. The function definition should therefore not receive a reference to 'self' as
    its first parameter.

    The decorator takes a number of optional keyword arguments:

    - `verbose` (False): if True, every time the attribute is recomputed, the instance method 
        '_caching_callback' is called. This does not happen when the attribute is retrieved from
        the cache.

    - `settable (False)`: if True, the attribute can be manually set using __setattr__ 
        (or via = ). If False, an AttributeError is raised on an attempted set.

    - `cskip`: under some conditions, certain arguments need not be used in evaluation
        of the attribute. This keyword provides a shortcut to prevent the retrieval of these 
        arguments under those conditions. `cskip` should be a list of triples
        `(arg1, value, arg2)` or `(arg1, value, arglist)`, where arguments are input as strings.
        If `arg1 == value`, then `arg2` (or the arguments in `arglist`) are not fetched, and 
        rather the function is evaluated with these latter arguments set to None. 

        Note that in the present implementation, if an argument appears in both the left
        hand side of a triple, and the right hand side of another triple, a ValueError is raised.

    """

    def __init__( self, func = None, verbose = False, settable = False, cskip = None ):
        self.verbose = verbose
        self.settable = settable

        # skipping: check arguments
        if cskip is None:
            cskip = []
        elif isinstance( cskip, tuple ):
            cskip = [cskip]
        elif isinstance( cskip, list ):
            pass
        else:
            raise ValueError('`cskip` should be a tuple or list of tuples')
        self.cskip = cskip if cskip is not None else []
        skip_arg1s = []
        skip_arg2s = []
        for i, cs in enumerate( self.cskip ):
            # cast the last element of each triple as a list
            if isinstance( cs[2], str ):
                self.cskip[i] = (cs[0], cs[1], [cs[2]])
            elif isinstance( cs[2], list ):
                pass
            else:
                raise TypeError('`%s` is not a string or list' % str(cs[2]))
            # accumulate arguments
            skip_arg1s += [cs[0]]
            skip_arg2s += cs[2]
        # check that no arguments appears on the left and right hand side of the skips
        invalid_args = set(skip_arg1s).intersection(skip_arg2s)
        if len(invalid_args) > 0:
            raise ValueError( '`%s` appears on both the LHS and RHS of a conditional skip' 
                    % invalid_args.pop() )

        # call
        if callable( func ):
            self.__call__( func )
        elif func is not None:
            raise TypeError, 'expected callable but got {} instead'.format( type( func ) )

    def __call__( self, func ):
        # imports
        from inspect import getargspec
        from functools import update_wrapper
        # get arguments of the function
        argspec = getargspec( func )
        # check remaining arguments
        if not ( argspec.varargs is None and
                 argspec.keywords is None and
                 argspec.defaults is None ):
            err_str = ( 'cannot handle default or variable arguments in cached ' + 
                'attribute dependencies' )
            raise NotImplementedError( err_str )

        # skipping
        parents = self.parents = argspec.args
        self.conditional_parents = set()
        # convert the triples into quadruples
        for i, cs in enumerate( self.cskip ):
            arg1, val, arglist = cs
            # check that all args in the cskip triple are valid
            args_to_check = arglist + [arg1]
            for a in args_to_check:
                if a not in parents:
                    raise ValueError('argument in cskip not a function argument: %s' % a)
            # add to conditional parents
            self.conditional_parents = self.conditional_parents.union(arglist)
            # indexes
            argidxs = [parents.index(a) for a in arglist]
            # save
            self.cskip[i] = (arg1, val, arglist, argidxs)
        # notes
        self.unconditional_parents = set([
                p for p in parents if p not in self.conditional_parents])
        self.check_for_skip = len(self.conditional_parents) > 0

        # finish wrapper
        self.func = func
        update_wrapper( self, func )
        return self

    def __get__( self, inst, cls ):
        if inst is None:
            return self
        # try get from cache, else compute
        try:
            return inst._cache[ self.__name__ ]
        except KeyError:
            # skip if required
            if self.check_for_skip:
                idxs_to_skip = [
                        cs[-1] for cs in self.cskip if getattr( inst, cs[0] ) == cs[1] ]
                if len( idxs_to_skip ) == 0:
                    idxs_to_skip += [[]]
                idxs_to_skip = reduce( lambda x, y: x+y, idxs_to_skip ) # flatten
                to_get = [True] * len(self.parents)
                for i in idxs_to_skip:
                    to_get[i] = False
                args = [ getattr( inst, a ) if q else None for q, a in zip( to_get, self.parents ) ]
            else:
                args = [ getattr( inst, a ) for a in self.parents ]
            # if we are skipping any arguments, set the others to None
            # evaluate the function
            if self.verbose:
                inst._caching_callback( name = self.__name__, arguments = args )
            value = self.func( *args )
            inst._cache[ self.__name__ ] = value
            return value

    def __set__( self, inst, value ):
        if self.settable:
            inst._cache[ self.__name__ ] = value
            inst._clear_descendants( self.__name__ )
        else:
            raise AttributeError, 'cached attribute {}.{} is not settable'.format(
                inst.__class__.__name__, self.__name__ )

    def __delete__( self, inst ):
        inst._cache.pop( self.__name__, None )
        inst._clear_descendants( self.__name__ )


class coupled( object ):

    def __init__( self, instname, attrname ):
        self.instname = instname
        self.attrname = attrname

    def __get__( self, inst, cls ):
        if inst is None:
            return self
        return getattr( getattr( inst, self.instname ), self.attrname )


class coupling( object ):

    def __get__( self, inst, cls ):
        if inst is None:
            return self
        return self.other

    def __set__( self, inst, other ):
        self.other = other
        cls = type( inst )
        for n in dir( cls ):
            m = getattr( cls, n )
            try:
                if getattr( cls, m.instname ) is not self:
                    continue
            except AttributeError:
                continue
            proxylist = other._proxies.get( m.attrname, [] )
            proxylist.append( ( inst, n ) )
            other._proxies[ m.attrname ] = proxylist


class _MetaAutoCacher( type ):

    def __init__( self, clsname, bases, dct ):
        super( _MetaAutoCacher, self ).__init__( clsname, bases, dct )

        # find all class members that are cached methods, store their parents
        # TODO: is it better to check for subclasses of 'cached' here?
        # there might be name conflicts with other class members that have a 'parents' attribute.
        # otoh, it is considered bad style to rely on explicit inheritance
        parents = {}
        for n in dir( self ):
            m = getattr( self, n )
            try:
                parents[ m.__name__ ] = m.parents
            except AttributeError:
                pass

        # collect descendants for each attribute that at least one cached attribute depends upon
        descendants = {}
        computing = set()
        def collect_descendants( name ):
            try:
                return descendants[ name ]
            except KeyError:
                if name in computing:
                    raise ValueError, 'circular dependency in cached attribute {}'.format( name )
                # direct descendants
                direct = [ c for c, p in parents.iteritems() if name in p ]
                des = set( direct )
                # update set with recursive descendants
                computing.add( name )
                for d in direct:
                    des.update( collect_descendants( d ) )
                computing.remove( name )
                descendants[ name ] = tuple( des )
                return des
        for n in set( p for par in parents.itervalues() for p in par ):
            collect_descendants( n )

        # collect ancestors for each cached attribute
        ancestors = {}
        def collect_ancestors( name ):
            try:
                return ancestors[ name ]
            except KeyError:
                # direct ancestors
                try:
                    direct = parents[ name ]
                except KeyError:
                    return ()
                anc = set( direct )
                # update set with recursive ancestors
                for d in direct:
                    anc.update( collect_ancestors( d ) )
                ancestors[ name ] = tuple( anc )
                return anc
        for n in parents.iterkeys():
            collect_ancestors( n )

        self._descendants = descendants
        self._ancestors = ancestors


class AutoCacher( object ):
    """Class with cached attributes.

    A class inheriting from 'AutoCacher' maintains a cache of attributes defined through the
    decorator 'cached', and keeps track of their dependencies in order to keep the cache up-to-date.

    Whenever a cached attribute is accessed for the first time, its value is computed according
    to its function definition, and then stored in the cache. On subsequent accesses, the cached
    value is returned, unless any of its dependencies (the parameters of the function definition)
    have changed. Example:

    class Objective( AutoCacher ):

       @cached
       def f( x, y ):
           return 2 * x + y

       @cached
       def g( f, y, z ):
           return f / y + z

    obj = Objective( x = 3, y = 4 )

    # this evaluates f and caches the result
    print obj.f
    obj.z = 3
    # this evaluates g using cached f
    print obj.g
    # this clears the cached values of f and g
    obj.x = 8
    # this reevaluates f and g
    print obj.g
    """

    __metaclass__ = _MetaAutoCacher

    def _clear_proxies( self, name ):
        for inst, attr in self._proxies.get( name, () ):
            inst._clear_descendants( attr )

    def _clear_descendants( self, name ):
        self._clear_proxies( name )
        for d in self._descendants.get( name, () ):
            self._cache.pop( d, None )
            self._clear_proxies( d )

    def _update( self, **kwargs ):
        for k, v in kwargs.iteritems():
            setattr( self, k, v )

    def __new__( cls, *args, **kwargs ):
        new = super( AutoCacher, cls ).__new__( cls, *args, **kwargs )
        super( AutoCacher, new ).__setattr__( '_cache', {} )
        super( AutoCacher, new ).__setattr__( '_proxies', {} )
        return new

    def __init__( self, **kwargs ):
        self._update( **kwargs )

    def __setattr__( self, name, value ):
        super( AutoCacher, self ).__setattr__( name, value )
        self._clear_descendants( name )

    def __delattr__( self, name ):
        super( AutoCacher, self ).__delattr__( name )
        self._clear_descendants( name )

    def __getstate__( self ):
        state = self.__dict__.copy()
        state.update( _cache = {} )
        return state

    class _Function( object ):

        def __init__( self, inst, output, *inputs ):
            self.cacher = inst
            self.output = output
            self.inputs = inputs

        def __call__( self, *inputs, **kwargs ):
            args = { k: v for k, v in zip( self.inputs, inputs ) }
            args.update( kwargs )
            self.cacher._update( **args )
            return getattr( self.cacher, self.output )

    def function( self, output, *inputs ):
        """Return a convenience function with defined input and outputs attribute names."""
        return AutoCacher._Function( self, output, *inputs )

    class _CFunction( object ):

        def __init__( self, inst, equal, output, *inputs ):
            self.cacher = inst
            self.equal = equal
            self.output = output
            self.inputs = inputs

        def __call__( self, *inputs, **kwargs ):
            args = { k: v for k, v in zip( self.inputs, inputs ) }
            args.update( kwargs )
            changed = {}
            for k, v in args.iteritems():
                try:
                    curval = getattr( self.cacher, k )
                except AttributeError:
                    changed[ k ] = v
                    continue
                if not self.equal( curval, v ):
                    changed[ k ] = v
            self.cacher._update( **changed )
            return getattr( self.cacher, self.output )

    def cfunction( self, equal, output, *inputs ):
        """Return a comparing convenience function with defined input and outputs attribute names."""
        return AutoCacher._CFunction( self, equal, output, *inputs )

    def clear_cache( self ):
        """Clear the cache: all cached attributes will need to be recomputed on access."""
        self._cache.clear()

    def descendants( self, name ):
        return self._descendants[ name ]

    def ancestors( self, name ):
        return self._ancestors[ name ]

    @property
    def _cache_summary( self ):
        return self.__summarise__( self._cache )

    @property
    def _dict_summary( self ):
        d = self.__dict__.copy()
        del d['_cache']
        del d['_proxies']
        return self.__summarise__( d )

    def __summarise__( self, d ):
        if len( d ) == 0:
            return 
        max_len_k = max([len(str(k)) for k in d.keys()])
        for k in sorted( d.keys() ):
            k_str = str(k)
            k_str += ' ' * (max_len_k - len(k_str)) + ' : '
            v = d[k]
            if isinstance(v, list):
                k_str += '(%d) list' % len(v)
            elif isinstance(v, tuple):
                k_str += '(%d) tuple' % len(v)
            elif isinstance(v, ndarray):
                k_str += str(v.shape) + ' array'
            elif isinstance(v, dict):
                k_str += '(%d) dict' % len(v)
            else:
                k_str += str(v)
            print '    ', k_str

    def plot_cache_graph( self, leaf=None, root=None, filename=None,
            display=True, display_cmd='eog' ):
        # imports
        import pydot
        import subprocess
        from numpy.random import random_integers
        # styles
        cached_shape = 'ellipse'
        ordinary_shape = 'box'
        present_penwidth = 3
        #fontname = 'helvetica-bold'
        fontname = 'DejaVuSans'
        styles = {}
        styles['cached absent'] = {
                'fontname':fontname,
                'shape':cached_shape, 
                'style':'filled', 
                'fillcolor':"#ffeeee"}
        styles['cached present'] = { 
                'fontname':fontname,
                'shape':cached_shape, 
                'style':'filled', 
                'fillcolor':"#ff9999", 
                'penwidth':present_penwidth }
        styles['ordinary absent'] = { 
                'fontname':fontname,
                'shape':ordinary_shape, 
                'style':'filled', 
                'fillcolor':"#eeeeff" }
        styles['ordinary present'] = { 
                'fontname':fontname,
                'shape':ordinary_shape, 
                'style':'filled', 
                'fillcolor':"#9999ff", 
                'penwidth':present_penwidth }
        styles['method'] = { 
                'fontname':fontname,
                'shape':'trapezium', 
                'style':'filled', 
                'margin':0,
                'fillcolor':"#aaccaa", 
                'penwidth':present_penwidth }

        g = pydot.Dot( graph_name='dependencies', rankdir='TB', ranksep=1 )
        g.set_node_defaults( fixedsize='false' )

        # useful
        cls = self.__class__

        # list of all the cachable attributes
        cachable_attrs = self._ancestors.keys()
        # dictionary of parents
        parents = { a : getattr( cls, a ).parents for a in cachable_attrs }
        conditional_parents = { a : getattr( cls, a ).conditional_parents
                for a in cachable_attrs }

        # list of all the attributes
        attrs = set(cachable_attrs)
        for a in cachable_attrs:
            # add its parents
            for p in parents[a]:
                attrs.add(p)

        # node type
        node_types = {}
        for a in attrs:
            if a in cachable_attrs:
                if self._cache.has_key(a):
                    node_types[a] = 'cached present'
                else:
                    node_types[a] = 'cached absent'
            elif ( hasattr( cls, a ) 
                    and getattr(cls, a).__class__.__name__ == (
                        'instancemethod') ):
                node_types[a] = 'method'
            else:
                if self.__dict__.has_key(a) or hasattr( cls, a ):
                    node_types[a] = 'ordinary present'
                else:
                    node_types[a] = 'ordinary absent'

        # restrict to attr of interest
        if leaf is not None:
            cachable_attrs = [a for a in cachable_attrs 
                    if a in self.ancestors( leaf ) or a == leaf]
            attrs = set(cachable_attrs)
            for a in cachable_attrs:
                # add its parents
                for p in parents[a]:
                    attrs.add(p)
        
        if root is not None:
            cachable_attrs = [a for a in cachable_attrs 
                    if a in self.descendants( root ) or a == root]
            attrs = set(cachable_attrs + [root])
            N_extra_parents = { a : len([p for p in v if p not in attrs])
                    for a, v in parents.items() }
            parents = { a : [p for p in v if p in attrs]
                for a, v in parents.items() }
            conditional_parents = { a : [p for p in v if p in attrs]
                for a, v in conditional_parents.items() }

        # create nodes
        nodes = {}
        for a in attrs:
            # label
            al = a
            if root is not None:
                if a in cachable_attrs: 
                    if N_extra_parents[a] > 0:
                        al = '%s  [+%d]' % (a, N_extra_parents[a])
            # create node
            nodes[a] = pydot.Node( name=a, label=al, **styles[node_types[a]] )
            g.add_node( nodes[a] )

        # create edges
        for a in cachable_attrs:
            # what edges have been temporarily silenced
            if len( conditional_parents[a] ) > 0:
                # get the conditional skips
                cskip = getattr( cls, a ).cskip
                # narrow down to known status
                cskip = [ cs for cs in cskip if node_types[cs[0]] in 
                        ['ordinary present', 'cached present'] ]
                # what should be skipped
                to_skip = [ cs[2] for cs in cskip 
                        if getattr( self, cs[0] ) == cs[1] ] + [[]]
                to_skip = reduce( lambda x, y : x+y, to_skip )
            else:
                to_skip = []
            # plot the edges
            for p in parents[a]:
                e = pydot.Edge( nodes[p], nodes[a] )
                if node_types[p] in [
                        'cached present', 'ordinary present', 'method']:
                    try:
                        e.set_penwidth(2)
                    except AttributeError:
                        pass
                else:
                    e.set_color('#888888')
                if p in to_skip:
                    e.set_style('dotted')
                elif p in conditional_parents[a]:
                    e.set_style('dashed')
                g.add_edge( e )

        # save file
        if filename is None:
            filename = '/tmp/cache.%d.png' % random_integers(1e12)

        g.write_png(filename, prog='dot')

        # display
        if display:
            process = subprocess.Popen(
                    "%s %s" % (display_cmd, filename), 
                    shell=True, stdout=subprocess.PIPE)
            return process

    def csetattr( self, attr, val ):
        """ Set the attr to the val, only if it is different. """
        f = self.cfunction( array_equal, attr, attr )
        f(val)



""" 
--------------------
Example of skipping
--------------------

class X( AutoCacher ):

    @cached( cskip = [ ('a', True, 'b'), ('c', 2, ['d', 'e']) ] )
    def z(a, b, c, d, e, f, g, h, i):
        return (a, b, c, d, e, f, g, h, i)

x = X()
x.a = True
x.b = 'ooga'
x.c = 2
x.d = 'booga'
x.e = 'awooga'
x.f = False
x.g = 'eep'
x.h = True
x.i = 'meep'

print x.z
x.c = 3
print x.z

"""





"""
===============================
Both patterns at the same time
===============================
"""


class MetaAutoCacherAndReloader( _MetaAutoCacher, MetaAutoReloader ):

    def __init__( cls, name, bases, ns ):
        # prepare autocache
        _MetaAutoCacher.__init__( cls, name, bases, ns )

        # prepare autoreload
        cls.__instance_refs__ = []
        f = inspect.currentframe().f_back
        for d in [f.f_locals, f.f_globals]:
            if name in d:
                old_class = d[name]
                for instance in old_class.__instances__():
                    instance.__change_class__(cls)
                    cls.__instance_refs__.append(weakref.ref(instance))
                for subcls in old_class.__subclasses__():
                    newbases = []
                    for base in subcls.__bases__:
                        if base is old_class:
                            newbases.append(cls)
                        else:
                            newbases.append(base)
                    subcls.__bases__ = tuple(newbases)
                break


class AutoCacherAndReloader( AutoReloader, AutoCacher ):

    __metaclass__ = MetaAutoCacherAndReloader

