#constant(obj, o0).
#constant(obj, o3).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o8).

#modeh(d1(var(obj))).
#modeh(d2(var(obj))).
#modeh(d4(var(obj))).
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
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d1(o5), d2(o5), d4(o5), d8(o5)
}, {
  d1(o0), d1(o8), d2(o0), d2(o3), d2(o6), d2(o8), d4(o3), d4(o6), d8(o0), d8(o3), d8(o6), d8(o8)
}, {
  d0(o3).
  d0(o6).
  d1(o3).
  d1(o6).
  d4(o0).
  d4(o8).
  d5(o5).
  d7(o5).
  d9(o5).
}).