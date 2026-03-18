#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o5).

#modeh(d0(var(obj))).
#modeh(d3(var(obj))).
#modeh(d6(var(obj))).
#modeh(d7(var(obj))).
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
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d0(o2), d3(o3), d3(o5), d6(o3), d6(o5), d7(o2), d9(o3)
}, {
  d0(o0), d0(o3), d0(o5), d3(o0), d3(o2), d6(o0), d6(o2), d7(o0), d7(o3), d7(o5), d9(o0), d9(o2)
}, {
  d1(o3).
  d1(o5).
  d2(o0).
  d4(o2).
  d8(o0).
  d9(o5).
}).