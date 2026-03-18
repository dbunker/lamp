#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o5).
#constant(obj, o7).
#constant(obj, o9).

#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d5(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d1(o3), d3(o0), d3(o3), d5(o0)
}, {
  d1(o2), d1(o5), d1(o7), d3(o7), d3(o9), d5(o2), d5(o5), d5(o9)
}, {
  d1(o0).
  d1(o9).
  d2(o3).
  d3(o2).
  d3(o5).
  d4(o0).
  d4(o3).
  d5(o3).
  d5(o7).
}).