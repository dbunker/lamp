#constant(obj, o0).
#constant(obj, o3).

#modeh(d1(var(obj))).
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
  d1(o0), d1(o3), d2(o0), d2(o3)
}, {
}, {
  d0(o0).
  d0(o3).
  d3(o0).
  d3(o3).
}).