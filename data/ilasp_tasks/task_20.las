#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).

#modeh(d2(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).

#pos(eg1, {
  d2(o1)
}, {
  d2(o0), d2(o2)
}, {
  d0(o2).
  d0(o3).
  d1(o0).
  d2(o3).
  d3(o1).
}).