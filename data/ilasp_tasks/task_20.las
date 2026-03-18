#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o4).

#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
#modeh(d5(var(obj))).
#modeh(d7(var(obj))).

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
  d2(o1), d3(o0), d5(o1), d5(o2), d7(o0)
}, {
  d3(o1), d3(o2), d3(o4), d5(o0), d7(o1), d7(o2), d7(o4)
}, {
  d2(o0).
  d2(o2).
  d2(o4).
  d4(o4).
  d5(o4).
  d6(o4).
  d8(o0).
  d8(o4).
  d9(o1).
}).