#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o5).

#modeh(d5(var(obj))).
#modeh(d7(var(obj))).
#modeh(d8(var(obj))).
#modeh(d9(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d5(o2), d7(o2), d8(o2), d9(o2)
}, {
  d5(o1), d5(o3), d5(o5), d7(o1), d8(o1), d8(o3), d8(o5), d9(o1), d9(o3), d9(o5)
}, {
  d0(o2).
  d2(o5).
  d4(o1).
  d4(o2).
  d7(o3).
  d7(o5).
}).