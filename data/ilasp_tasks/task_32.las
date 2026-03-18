#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o5).

#modeh(d0(var(obj))).
#modeh(d4(var(obj))).
#modeh(d6(var(obj))).
#modeh(d7(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
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
  d0(o3), d4(o3), d6(o3), d7(o3)
}, {
  d0(o2), d0(o5), d4(o2), d4(o5), d6(o2), d7(o5)
}, {
  d5(o3).
  d5(o5).
  d6(o5).
  d7(o2).
  d8(o3).
  d9(o5).
}).