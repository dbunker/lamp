#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o9).

#modeh(d0(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
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

#pos(eg1, {
  d0(o0), d0(o4), d3(o4), d4(o4)
}, {
  d0(o1), d0(o3), d0(o5), d0(o6), d0(o9), d3(o0), d3(o1), d3(o3), d3(o5), d3(o9), d4(o3), d4(o5), d4(o6), d4(o9)
}, {
  d1(o4).
  d3(o6).
  d4(o0).
  d4(o1).
  d6(o9).
  d7(o5).
  d8(o0).
  d8(o3).
  d8(o4).
}).