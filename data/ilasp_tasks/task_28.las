#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d8(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d1(o1), d3(o1), d3(o2), d8(o1), d8(o2)
}, {
  d1(o0), d1(o3), d1(o5), d3(o0), d3(o3), d3(o4), d3(o5), d8(o0), d8(o3), d8(o4), d8(o5)
}, {
  d1(o2).
  d1(o4).
  d2(o1).
  d2(o2).
  d2(o5).
  d6(o0).
  d7(o3).
  d9(o1).
  d9(o2).
}).