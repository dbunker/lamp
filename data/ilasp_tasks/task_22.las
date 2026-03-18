#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d2(var(obj))).
#modeh(d7(var(obj))).
#modeh(d8(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
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
  d2(o5), d7(o4), d8(o4)
}, {
  d2(o2), d2(o3), d2(o4), d7(o0), d7(o2), d7(o3), d8(o0), d8(o2), d8(o3)
}, {
  d0(o3).
  d0(o4).
  d1(o5).
  d2(o0).
  d4(o2).
  d5(o4).
  d6(o5).
  d7(o5).
  d8(o5).
}).