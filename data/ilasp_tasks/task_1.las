#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o7).
#constant(obj, o8).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
#modeh(d5(var(obj))).
#modeh(d6(var(obj))).
#modeh(d9(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
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
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d0(o4), d0(o8), d1(o3), d1(o4), d1(o8), d5(o2), d5(o3), d6(o8), d9(o2), d9(o3)
}, {
  d0(o0), d0(o2), d0(o3), d1(o0), d1(o2), d1(o7), d5(o0), d5(o4), d5(o7), d5(o8), d6(o0), d6(o2), d6(o3), d6(o4), d6(o7), d9(o0), d9(o4)
}, {
  d0(o7).
  d2(o2).
  d2(o3).
  d3(o4).
  d4(o2).
  d7(o7).
  d8(o0).
  d9(o7).
  d9(o8).
}).