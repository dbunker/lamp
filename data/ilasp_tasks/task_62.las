#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).

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
  d1(o3), d1(o4), d2(o2), d2(o3), d2(o4)
}, {
  d1(o1)
}, {
  d0(o1).
  d1(o2).
  d2(o1).
  d3(o3).
  d3(o4).
}).