#constant(obj, o1).
#constant(obj, o3).

#modeh(d0(var(obj))).
#modeh(d3(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).

#pos(eg1, {
  d0(o1), d0(o3), d3(o1), d3(o3)
}, {
}, {
  d1(o1).
  d1(o3).
  d2(o1).
  d2(o3).
}).