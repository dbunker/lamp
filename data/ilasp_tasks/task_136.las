#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).

#modeh(d4(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).

#pos(eg1, {
  d4(o0)
}, {
  d4(o3), d4(o4)
}, {
  d1(o0).
  d1(o4).
  d2(o0).
  d3(o3).
  d4(o2).
}).