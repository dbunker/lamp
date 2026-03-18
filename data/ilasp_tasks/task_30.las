#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d2(var(obj))).
#modeh(d6(var(obj))).
#modeh(d7(var(obj))).
#modeh(d8(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d2(o1), d6(o1), d7(o5), d8(o3)
}, {
  d2(o0), d2(o3), d2(o4), d2(o5), d6(o0), d6(o3), d6(o4), d6(o5), d7(o0), d7(o1), d7(o4), d8(o0), d8(o1), d8(o4), d8(o5)
}, {
  d0(o1).
  d0(o5).
  d3(o4).
  d5(o1).
  d7(o3).
  d9(o0).
}).