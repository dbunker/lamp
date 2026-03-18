#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).
#modeh(d5(var(obj))).
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
  d2(o4), d3(o4), d4(o4), d5(o4), d7(o4), d9(o4)
}, {
  d2(o3), d2(o5), d3(o3), d3(o5), d4(o3), d4(o5), d5(o5), d7(o3), d7(o5), d9(o3), d9(o5)
}, {
  d0(o4).
  d1(o4).
  d5(o3).
  d6(o4).
  d6(o5).
  d8(o4).
}).