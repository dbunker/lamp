#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o4).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
#modeh(d5(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d0(o0), d0(o4), d1(o1), d5(o2)
}, {
  d0(o1), d0(o2), d1(o0), d1(o4), d5(o4)
}, {
  d1(o2).
  d2(o0).
  d2(o4).
  d5(o0).
  d5(o1).
}).