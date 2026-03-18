#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o7).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d6(var(obj))).
#modeh(d9(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
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
  d6(o1), d6(o2), d9(o3)
}, {
  d6(o3), d6(o7), d6(o8), d6(o9), d9(o1), d9(o2), d9(o7), d9(o8), d9(o9)
}, {
  d0(o1).
  d0(o2).
  d1(o3).
  d5(o1).
  d5(o9).
  d7(o3).
  d7(o8).
  d8(o2).
  d8(o7).
}).