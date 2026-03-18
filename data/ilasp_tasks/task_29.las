#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).

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
  d0(o1), d1(o1), d2(o1), d3(o1), d4(o1)
}, {
  d0(o0), d0(o2), d0(o3), d1(o0), d1(o2), d1(o3), d2(o0), d2(o3), d3(o0), d3(o2), d4(o2)
}, {
  d2(o2).
  d3(o3).
  d4(o0).
  d4(o3).
  d5(o1).
  d6(o1).
  d7(o1).
  d8(o1).
  d9(o1).
}).