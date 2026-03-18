#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o6).
#constant(obj, o7).

#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d6(var(obj))).
#modeh(d7(var(obj))).
#modeh(d8(var(obj))).

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
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d1(o6), d3(o0), d3(o4), d3(o7), d6(o0), d7(o0), d7(o4), d7(o6), d7(o7), d8(o3), d8(o6)
}, {
  d1(o0), d1(o2), d1(o3), d1(o4), d1(o7), d3(o2), d3(o3), d3(o6), d6(o2), d6(o6), d8(o0), d8(o2), d8(o4), d8(o7)
}, {
  d2(o0).
  d2(o3).
  d4(o6).
  d5(o0).
  d6(o3).
  d6(o4).
  d6(o7).
  d7(o2).
  d7(o3).
}).